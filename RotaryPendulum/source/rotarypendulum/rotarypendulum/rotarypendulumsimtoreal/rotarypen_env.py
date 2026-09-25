"""Synchronous native MuJoCo rollouts with one independent model per environment."""
from dataclasses import asdict
import math

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from .mdp import events, observations, rewards, terminations
from .mdp.learned_motor_action import LearnedMotorAction, RAD_S_TO_RPM
from .rotarypen_cfg import MJCF_PATH
from .rotarypen_env_cfg import RotarypenSim2RealEnvCfg


class RotaryPendulumEnv:
    def __init__(self, cfg: RotarypenSim2RealEnvCfg, device="cpu", randomization=True):
        self.cfg, self.device, self.randomization = cfg, device, randomization
        self.num_envs = cfg.scene.num_envs
        if not 1 <= self.num_envs < 4000:
            raise ValueError("num_envs must be 1..3999")
        if abs(cfg.sim.dt * 300 - 1) > 1e-9 and abs(cfg.sim.dt * 600 - 1) > 1e-9:
            raise ValueError("physics dt must be 1/300 or 1/600")
        if abs(cfg.sim.dt * cfg.sim.decimation - 1 / 30) > 1e-9:
            raise ValueError("policy rate must remain 30 Hz")
        self.pid_stride = round(0.02 / cfg.sim.dt)
        self.models = [mujoco.MjModel.from_xml_path(str(MJCF_PATH)) for _ in range(self.num_envs)]
        for model in self.models:
            model.opt.timestep = cfg.sim.dt
        self.data = [mujoco.MjData(model) for model in self.models]
        self.nominal_mass = [model.body_mass.copy() for model in self.models]
        self.nominal_inertia = [model.body_inertia.copy() for model in self.models]
        self.nominal_damping = [model.dof_damping.copy() for model in self.models]
        self.nominal_friction = [model.dof_frictionloss.copy() for model in self.models]
        self.motor = LearnedMotorAction(cfg.actions.motor, self.num_envs, device)
        self.rng = np.random.default_rng(cfg.seed)
        self.torch_rng = torch.Generator(device=device).manual_seed(cfg.seed)
        self.max_episode_length = round(cfg.sim.episode_length_s * 30)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.common_step_counter = 0
        self.physics_counter = 0
        self.num_actions = 1
        self.obs_history = np.zeros((self.num_envs, cfg.observations.history_length, 6), dtype=np.float32)
        self.action = np.zeros(self.num_envs, dtype=np.float32)
        self.previous_action = self.action.copy()
        self.last_reference_rpm = np.zeros(self.num_envs)
        self.last_force = np.zeros(self.num_envs)
        self.reset()

    def seed(self, seed):
        self.cfg.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.torch_rng.manual_seed(seed)
        return seed

    def state(self):
        return np.array([d.qpos.copy() for d in self.data]), np.array([d.qvel.copy() for d in self.data])

    def _randomize_model(self, i):
        model = self.models[i]
        model.body_mass[:] = self.nominal_mass[i]
        model.body_inertia[:] = self.nominal_inertia[i]
        model.dof_damping[:] = self.nominal_damping[i]
        model.dof_frictionloss[:] = self.nominal_friction[i]
        if self.randomization:
            lo, hi = self.cfg.scene.moving_mass_scale_range
            for body_id in (2, 3):
                scale = self.rng.uniform(lo, hi)
                model.body_mass[body_id] *= scale
                model.body_inertia[body_id] *= scale
            lo, hi = self.cfg.scene.joint_damping_scale_range
            model.dof_damping[1] *= self.rng.uniform(lo, hi)
            lo, hi = self.cfg.scene.joint_friction_range
            model.dof_frictionloss[:] = self.rng.uniform(lo, hi, 2)
        mujoco.mj_setConst(model, self.data[i])

    def reset_ids(self, ids, start=None):
        ids = list(ids)
        if not ids:
            return
        q, qd = events.reset_swingup_balance(self.rng, len(ids), self.cfg.events.reset_arm.params, self.common_step_counter, start)
        self.motor.reset(ids, self.torch_rng)
        for j, i in enumerate(ids):
            self._randomize_model(i)
            mujoco.mj_resetData(self.models[i], self.data[i])
            self.data[i].qpos[:] = q[j]
            self.data[i].qvel[:] = qd[j]
            mujoco.mj_forward(self.models[i], self.data[i])
        self.episode_length_buf[ids] = 0
        self.action[ids] = 0
        self.previous_action[ids] = 0
        frame = observations.frame(*self.state(), self.action)
        self.obs_history[ids] = frame[ids, None, :]

    def reset(self, start=None):
        self.reset_ids(range(self.num_envs), start=start)
        return self.get_observations(), {}

    def get_observations(self):
        critic = torch.as_tensor(self.obs_history.reshape(self.num_envs, -1).copy(), device=self.device)
        policy = critic.clone()
        if self.cfg.observations.actor_noise:
            policy += (torch.rand(policy.shape, device=self.device, generator=self.torch_rng) * 2 - 1) * self.cfg.observations.actor_noise
        return TensorDict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        raw = actions[:, 0].detach().cpu().numpy().copy()
        if raw.shape != (self.num_envs,) or not np.isfinite(raw).all():
            raise ValueError("actions must be finite [num_envs,1]")
        self.previous_action[:] = self.action
        self.action[:] = raw
        self.motor.command(actions)
        for _ in range(self.cfg.sim.decimation):
            if self.physics_counter % self.pid_stride == 0:
                _, qd = self.state()
                self.motor.controller_update(torch.as_tensor(qd[:, :1] * RAD_S_TO_RPM, dtype=torch.float32, device=self.device))
            phase = (self.physics_counter % self.pid_stride + 1) / self.pid_stride
            reference = self.motor.reference(phase).detach().cpu().numpy()[:, 0]
            self.last_reference_rpm[:] = reference
            for i, (model, data) in enumerate(zip(self.models, self.data)):
                data.ctrl[0] = reference[i] / RAD_S_TO_RPM
                mujoco.mj_step(model, data)
                self.last_force[i] = data.actuator_force[0]
            self.physics_counter += 1
        self.episode_length_buf += 1
        self.common_step_counter += 1
        q, qd = self.state()
        terminated = terminations.arm_safety(q, self.cfg.terminations.arm_safety.params.limit_deg) | ~np.isfinite(q).all(axis=1) | ~np.isfinite(qd).all(axis=1)
        truncated = self.episode_length_buf.cpu().numpy() >= self.max_episode_length
        reward = rewards.weighted(q, qd, self.action, self.previous_action, terminated, self.cfg.rewards, self.cfg.sim.decimation * self.cfg.sim.dt)
        frame = observations.frame(q, qd, self.action)
        self.obs_history = np.roll(self.obs_history, -1, axis=1)
        self.obs_history[:, -1, :] = frame
        final_observation = self.obs_history.reshape(self.num_envs, -1).copy()
        done = terminated | truncated
        if done.any():
            self.reset_ids(np.flatnonzero(done))
        obs = self.get_observations()
        extras = {
            "time_outs": torch.as_tensor(truncated & ~terminated, dtype=torch.bool, device=self.device),
            "final_observation": torch.as_tensor(final_observation, dtype=torch.float32, device=self.device),
            "terminated": torch.as_tensor(terminated, device=self.device),
            "truncated": torch.as_tensor(truncated, device=self.device),
        }
        return obs, torch.as_tensor(reward, device=self.device), torch.as_tensor(done, device=self.device, dtype=torch.long), extras

    def close(self):
        self.data.clear()
        self.models.clear()
