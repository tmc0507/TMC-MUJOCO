"""User-editable MuJoCo environment settings, grouped like the Isaac task."""
from dataclasses import dataclass, field, asdict
from .mdp.learned_motor_action import MotorCfg


@dataclass
class SceneCfg:
    num_envs: int = 32
    moving_mass_scale_range: tuple[float, float] = (0.95, 1.05)
    joint_damping_scale_range: tuple[float, float] = (0.85, 1.15)
    joint_friction_range: tuple[float, float] = (0.0, 0.0008)


@dataclass
class ActionsCfg:
    motor: MotorCfg = field(default_factory=MotorCfg)


@dataclass
class ObservationsCfg:
    history_length: int = 8
    actor_noise: float = 0.001


@dataclass
class ResetParams:
    upright_fraction: float = 0.8
    uniform_fraction: float = 0.0
    curriculum_steps: int = 12800


@dataclass
class ResetCfg:
    params: ResetParams = field(default_factory=ResetParams)


@dataclass
class EventsCfg:
    reset_arm: ResetCfg = field(default_factory=ResetCfg)


@dataclass
class RewardTerm:
    weight: float


@dataclass
class RewardsCfg:
    upright_progress: RewardTerm = field(default_factory=lambda: RewardTerm(2.0))
    upright_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(12.0))
    stable: RewardTerm = field(default_factory=lambda: RewardTerm(4.0))
    energy: RewardTerm = field(default_factory=lambda: RewardTerm(2.0))
    arm_position: RewardTerm = field(default_factory=lambda: RewardTerm(-0.5))
    pendulum_vel: RewardTerm = field(default_factory=lambda: RewardTerm(-0.35))
    arm_center: RewardTerm = field(default_factory=lambda: RewardTerm(-0.50))
    arm_vel: RewardTerm = field(default_factory=lambda: RewardTerm(-0.05))
    action_rate: RewardTerm = field(default_factory=lambda: RewardTerm(-0.1))
    action_mag: RewardTerm = field(default_factory=lambda: RewardTerm(-0.002))
    alive: RewardTerm = field(default_factory=lambda: RewardTerm(0.02))
    termination_penalty: RewardTerm = field(default_factory=lambda: RewardTerm(-10.0))


@dataclass
class SafetyParams:
    limit_deg: float = 125.0


@dataclass
class SafetyCfg:
    params: SafetyParams = field(default_factory=SafetyParams)


@dataclass
class TerminationsCfg:
    arm_safety: SafetyCfg = field(default_factory=SafetyCfg)


@dataclass
class SimCfg:
    dt: float = 1 / 300
    decimation: int = 10
    episode_length_s: float = 10.0


@dataclass
class RotarypenSim2RealEnvCfg:
    scene: SceneCfg = field(default_factory=SceneCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    events: EventsCfg = field(default_factory=EventsCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    seed: int = 42


def apply_overrides(env, agent, overrides):
    """Apply dotted assignments; unknown paths and incompatible values are errors."""
    for item in overrides:
        path, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected key=value: {item}")
        parts = path.split(".")
        if parts[0] not in ("env", "agent"):
            raise ValueError(f"Override must start env. or agent.: {path}")
        obj = env if parts[0] == "env" else agent
        for part in parts[1:-1]:
            if not hasattr(obj, part):
                raise ValueError(f"Unknown config key: {path}")
            obj = getattr(obj, part)
        key = parts[-1]
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {path}")
        old = getattr(obj, key)
        if isinstance(old, bool):
            if value.lower() not in ("true", "false"):
                raise ValueError(f"Expected boolean for {path}")
            new = value.lower() == "true"
        elif isinstance(old, int):
            new = int(value)
        elif isinstance(old, float):
            new = float(value)
        elif isinstance(old, str):
            new = value
        else:
            raise ValueError(f"Unsupported override type: {path}")
        setattr(obj, key, new)
