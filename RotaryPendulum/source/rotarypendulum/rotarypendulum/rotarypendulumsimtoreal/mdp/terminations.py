import numpy as np


def arm_safety(q, limit_deg=125):
    arm = (q[:, 0] + np.pi) % (2 * np.pi) - np.pi
    return np.abs(arm) > np.deg2rad(limit_deg)
