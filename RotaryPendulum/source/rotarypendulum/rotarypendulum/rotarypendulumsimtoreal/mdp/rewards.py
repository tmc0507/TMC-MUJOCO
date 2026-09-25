import numpy as np


def terms(q, qd, action, previous_action, terminated):
    arm = (q[:, 0] + np.pi) % (2 * np.pi) - np.pi
    theta = (q[:, 1] + np.pi) % (2 * np.pi) - np.pi
    error = (q[:, 1] + 2 * np.pi) % (2 * np.pi) - np.pi
    near = (np.abs(theta) >= np.deg2rad(145)).astype(float)
    kinetic = 0.5 * 9.24e-5 * qd[:, 1] ** 2 / (0.024 * 9.81 * 0.04963)
    energy = np.maximum(0, 4 - (kinetic + 1 - np.cos(theta) - 2) ** 2)
    quality = np.exp(-(error / 0.25) ** 2) / (1 + (qd[:, 1] / 1.5) ** 2 + (qd[:, 0] / 1.5) ** 2 + (arm / 0.8) ** 2)
    return {
        "upright_progress": 0.5 * (1 - np.cos(theta)),
        "upright_bonus": quality,
        "stable": ((np.abs(error) < np.deg2rad(10)) & (np.abs(qd) < 2).all(axis=1)).astype(float),
        "energy": energy,
        "arm_position": (arm / 2.18) ** 2,
        "pendulum_vel": near * (qd[:, 1] / 3) ** 2,
        "arm_center": near * (arm / 2.6) ** 2,
        "arm_vel": (qd[:, 0] / 25) ** 2,
        "action_rate": (action - previous_action) ** 2,
        "action_mag": action ** 2,
        "alive": 1 - terminated.astype(float),
        "termination_penalty": terminated.astype(float),
    }


def weighted(q, qd, action, previous_action, terminated, cfg, policy_dt):
    values = terms(q, qd, action, previous_action, terminated)
    return (sum(getattr(cfg, name).weight * value for name, value in values.items()) * policy_dt).astype(np.float32)
