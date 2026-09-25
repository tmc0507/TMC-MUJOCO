import numpy as np


def reset_swingup_balance(rng, count, params, common_step_counter, start=None):
    q = np.zeros((count, 2))
    qd = rng.uniform(-0.15, 0.15, (count, 2))
    q[:, 0] = rng.uniform(-0.15, 0.15, count)
    sample = rng.random(count)
    up = sample < params.upright_fraction
    if start == "down":
        up[:] = False
    elif start == "upright":
        up[:] = True
    width = 0.15 + (np.pi - 0.15) * min(common_step_counter / params.curriculum_steps, 1) if params.curriculum_steps > 0 else 0.15
    q[up, 1] = np.pi + rng.uniform(-width, width, up.sum())
    q[~up, 1] = rng.uniform(-0.15, 0.15, (~up).sum())
    uniform = (sample >= params.upright_fraction) & (sample < params.upright_fraction + params.uniform_fraction)
    if start is None:
        q[uniform, 1] = rng.uniform(-np.pi, np.pi, uniform.sum())
    return q, qd
