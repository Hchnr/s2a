"""198 fixed282 current basis plus39 measured contact-history features."""

import numpy as np

from s2a.joint_wrench_condition_forward import design as current_design


def design(data, parameters):
    current = current_design(data, parameters)
    history = np.asarray(data["x_finger_contact_history"], dtype=float)
    mean = np.asarray(parameters["finger_history_mean"], dtype=float)
    scale = np.asarray(parameters["finger_history_scale"], dtype=float)
    if (
        history.shape != (len(current), 39)
        or mean.shape != (39,)
        or scale.shape != (39,)
        or not all(np.isfinite(v).all() for v in (history, mean, scale))
        or np.any(scale <= 0)
    ):
        raise ValueError("Explicit finite39 current/past history and fixed positive training scales required")
    z = (history - mean) / scale
    blocks = []
    for index in range(4):
        block = current[:, 282 * index : 282 * (index + 1)]
        blocks.append(np.c_[block, z * block[:, :1]])
    return np.hstack(blocks)
