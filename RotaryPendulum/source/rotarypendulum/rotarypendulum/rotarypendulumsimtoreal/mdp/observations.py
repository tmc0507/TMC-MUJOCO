import numpy as np


def frame(q, qd, previous_action):
    arm = (q[:, 0] + np.pi) % (2 * np.pi) - np.pi
    return np.stack((arm / 2.6, np.sin(q[:, 1]), np.cos(q[:, 1]), qd[:, 0] / 25, qd[:, 1] / 25, previous_action), axis=-1).astype(np.float32)
