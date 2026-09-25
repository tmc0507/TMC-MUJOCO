"""Frozen Hybrid TCN and the source firmware PID, without Isaac Lab imports."""

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.silu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = F.silu(y)
        y = self.dropout(y)
        return x + y


class TemporalMotorMember(nn.Module):
    def __init__(
        self,
        input_size: int,
        channels: int,
        levels: int,
        kernel_size: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(levels)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, max(16, channels // 2)),
            nn.SiLU(),
            nn.Linear(max(16, channels // 2), 1),
        )

    def forward(self, x_btf: torch.Tensor) -> torch.Tensor:
        x = x_btf.transpose(1, 2)  # [B,T,F] -> [B,F,T]
        x = F.silu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, :, -1])


class HybridMotorTCNInference(nn.Module):
    """Inference-only network matching the V3/V3.1 checkpoint architecture."""

    def __init__(
        self,
        history_len: int,
        channels: int,
        levels: int,
        ensemble_size: int,
        kernel_size: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.input_size = 5
        self.members = nn.ModuleList(
            [
                TemporalMotorMember(
                    input_size=self.input_size,
                    channels=int(channels),
                    levels=int(levels),
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
                for _ in range(int(ensemble_size))
            ]
        )

    def forward(self, x_btf: torch.Tensor) -> torch.Tensor:
        # [ensemble, batch, 1]
        return torch.stack([m(x_btf) for m in self.members], dim=0)


@dataclass
class MotorCfg:
    checkpoint_path: str = ""
    action_limit: float = 1.0
    target_rpm_limit: float = 2200.0
    pid_kp: float = 0.26925
    pid_ki: float = 1.45549
    pid_kd: float = 0.00287
    pwm_limit: float = 250.0
    encoder_cpr: int = 1024
    pid_dt_s: float = 0.02
    motor_d_nominal: float = 2.10e-4
    torque_limit_nominal_nm: float = 0.060
    motor_d_scale_range: tuple[float, float] = (0.60, 1.40)
    torque_limit_scale_range: tuple[float, float] = (0.70, 1.30)
    supply_voltage_scale_range: tuple[float, float] = (0.92, 1.05)
    pid_gain_scale_range: tuple[float, float] = (0.95, 1.05)
    policy_delay_frames_max: int = 1
    model_velocity_tracking: bool = True
    model_velocity_gain_nm_per_rad_s: float = 2.0
    model_velocity_torque_limit_nm: float = 3.0
    clamp_features_to_training_range: bool = True
    clamp_prediction_to_training_rpm: bool = True
    prediction_margin_rpm: float = 25.0


class LearnedMotorAction:
    def __init__(self, cfg: MotorCfg, num_envs: int, device: str = "cpu"):
        self.cfg, self.num_envs, self.device = cfg, num_envs, device
        path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else Path(__file__).resolve().parents[1] / "models/modelmotorvitual.pth"
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("format_version") != 3 or ckpt.get("model_type") != "HybridMotorTCN":
            raise ValueError("Expected frozen HybridMotorTCN format 3")
        if ckpt["model_config"]["history_len"] != 25 or abs(ckpt["nominal_dt"] - cfg.pid_dt_s) > 1e-7:
            raise ValueError("TCN history/dt mismatch")
        self.plant = HybridMotorTCNInference(**ckpt["model_config"]).to(device).eval()
        self.plant.load_state_dict(ckpt["model_state"], strict=True)
        self.plant.requires_grad_(False)
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.x_mean = t(ckpt["scaler_X"]["mean"]).view(1, 1, 5)
        self.x_scale = t(ckpt["scaler_X"]["scale"]).view(1, 1, 5)
        self.y_mean = t(ckpt["scaler_y"]["mean"][0])
        self.y_scale = t(ckpt["scaler_y"]["scale"][0])
        self.coef = t(ckpt["physics_coef"])
        self.feature_min = None if ckpt.get("feature_min") is None else t(ckpt["feature_min"]).view(1, 1, 5)
        self.feature_max = None if ckpt.get("feature_max") is None else t(ckpt["feature_max"]).view(1, 1, 5)
        self.rpm_min, self.rpm_max = float(ckpt["train_rpm_min"]), float(ckpt["train_rpm_max"])
        self.pwm_min, self.pwm_max = float(ckpt["train_pwm_min"]), float(ckpt["train_pwm_max"])
        if cfg.target_rpm_limit > min(abs(self.rpm_min), abs(self.rpm_max)) + 1:
            raise ValueError("Target RPM exceeds training envelope")
        self.reset_all()

    def reset_all(self):
        n = self.num_envs
        z = lambda: torch.zeros((n, 1), dtype=torch.float32, device=self.device)
        for name in ("raw_action", "prev_action", "target_rpm", "model_rpm", "predicted_rpm", "pwm", "effective_pwm", "integral", "last_error", "last_rpm", "last_pwm", "interval_start", "interval_end", "joint_rpm"):
            setattr(self, name, z())
        self.history = torch.zeros((n, 25, 5), dtype=torch.float32, device=self.device)
        self.history_initialized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.delay = torch.zeros((n, 1), dtype=torch.long, device=self.device)
        self.kp, self.ki, self.kd, self.supply = z(), z(), z(), z()
        self.reset(range(n))

    def reset(self, ids, generator=None):
        ids = list(ids)
        if not ids:
            return
        for name in ("raw_action", "prev_action", "target_rpm", "model_rpm", "predicted_rpm", "pwm", "effective_pwm", "integral", "last_error", "last_rpm", "last_pwm", "interval_start", "interval_end", "joint_rpm"):
            getattr(self, name)[ids] = 0
        self.history[ids] = 0
        self.history_initialized[ids] = False
        u = lambda lo, hi: torch.empty((len(ids), 1), device=self.device).uniform_(lo, hi, generator=generator)
        lo, hi = self.cfg.pid_gain_scale_range
        self.kp[ids] = self.cfg.pid_kp * u(lo, hi)
        self.ki[ids] = self.cfg.pid_ki * u(lo, hi)
        self.kd[ids] = self.cfg.pid_kd * u(lo, hi)
        lo, hi = self.cfg.supply_voltage_scale_range
        self.supply[ids] = u(lo, hi)
        self.delay[ids] = torch.randint(0, self.cfg.policy_delay_frames_max + 1, (len(ids), 1), device=self.device, generator=generator)

    def command(self, action: torch.Tensor):
        incoming = action[:, :1].clamp(-self.cfg.action_limit, self.cfg.action_limit)
        delayed = torch.where(self.delay > 0, self.prev_action, incoming)
        self.prev_action.copy_(incoming)
        self.raw_action.copy_(delayed)
        self.target_rpm.copy_(delayed / self.cfg.action_limit * self.cfg.target_rpm_limit)

    def pid(self, measured: torch.Tensor):
        cfg = self.cfg
        error = self.target_rpm - measured
        derivative = (error - self.last_error) / cfg.pid_dt_s
        before = self.kp * error + self.ki * self.integral + self.kd * derivative
        integrate = ((before > -cfg.pwm_limit) & (before < cfg.pwm_limit)) | ((before >= cfg.pwm_limit) & (error < 0)) | ((before <= -cfg.pwm_limit) & (error > 0))
        self.integral.copy_(torch.where(integrate, self.integral + error * cfg.pid_dt_s, self.integral))
        self.last_error.copy_(error)
        return torch.trunc((self.kp * error + self.ki * self.integral + self.kd * derivative).clamp(-cfg.pwm_limit, cfg.pwm_limit))

    @torch.inference_mode()
    def controller_update(self, joint_rpm: torch.Tensor):
        self.joint_rpm.copy_(joint_rpm)
        measured = self.model_rpm if self.cfg.model_velocity_tracking else joint_rpm
        pwm = self.pid(measured).clamp(max(self.pwm_min, -self.cfg.pwm_limit), min(self.pwm_max, self.cfg.pwm_limit))
        self.pwm.copy_(pwm)
        effective = torch.trunc(pwm * self.supply).clamp(max(self.pwm_min, -self.cfg.pwm_limit), min(self.pwm_max, self.cfg.pwm_limit))
        self.effective_pwm.copy_(effective)
        drpm = torch.where(self.history_initialized[:, None], measured - self.last_rpm, 0)
        dpwm = torch.where(self.history_initialized[:, None], effective - self.last_pwm, 0)
        feat = torch.cat((measured, effective, drpm, dpwm, torch.full_like(measured, self.cfg.pid_dt_s)), dim=1)
        fresh = ~self.history_initialized
        self.history[fresh] = feat[fresh, None, :].expand(-1, 25, -1)
        old = self.history_initialized
        self.history[old] = torch.roll(self.history[old], -1, dims=1)
        self.history[old, -1] = feat[old]
        self.history_initialized[:] = True
        self.last_rpm.copy_(measured)
        self.last_pwm.copy_(effective)
        hist = self.history
        if self.cfg.clamp_features_to_training_range and self.feature_min is not None:
            hist = torch.maximum(torch.minimum(hist, self.feature_max), self.feature_min)
        x = (hist - self.x_mean) / self.x_scale.clamp_min(1e-8)
        residual = self.plant(x).squeeze(-1) * self.y_scale + self.y_mean
        c = self.coef
        accel = c[0] + c[1] * measured + c[2] * effective + c[3] * torch.tanh(measured / 5) + c[4] * torch.tanh(effective / 20)
        next_rpm = measured + accel * self.cfg.pid_dt_s + residual.mean(dim=0, keepdim=False)[:, None]
        if self.cfg.clamp_prediction_to_training_rpm:
            next_rpm = next_rpm.clamp(self.rpm_min - self.cfg.prediction_margin_rpm, self.rpm_max + self.cfg.prediction_margin_rpm)
        self.interval_start.copy_(self.model_rpm)
        self.interval_end.copy_(next_rpm)
        self.predicted_rpm.copy_(next_rpm)
        self.model_rpm.copy_(next_rpm)

    def reference(self, phase: float):
        return self.interval_start + phase * (self.interval_end - self.interval_start)
