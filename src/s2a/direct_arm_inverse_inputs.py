"""207 causal current conditions and explicit generated next SITE pose for inversion."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import validate_transform
from s2a.joint_wrench_condition_forward import raw_condition


def inverse_condition(data, desired_site, *, gripper_command):
    current = validate_transform(np.asarray(data["x_site_pose"], dtype=float))
    desired = validate_transform(np.asarray(desired_site, dtype=float))
    if current.ndim != 3 or desired.shape != current.shape:
        raise ValueError("Aligned batches of current measured and generated desired SITE poses required")
    count = len(current)
    grip = np.asarray(gripper_command, dtype=float)
    if grip.shape != (count,) or not np.isfinite(grip).all() or np.any(np.abs(grip) > 1):
        raise ValueError("Explicit feasible current gripper commands required")
    sanitized = dict(data)
    # Target arm action and its derived controller goal must never be inverse inputs.
    sanitized["action_command"] = np.c_[np.zeros((count, 6)), grip]
    sanitized["action_goal_delta"] = np.zeros((count, 6))
    raw = raw_condition(sanitized)
    if raw.shape != (count, 89):
        raise ValueError("Fixed current89 input contract required")
    measured = np.c_[raw[:, :64], raw[:, 70:]]
    motion = desired[:, :3, 3] - current[:, :3, 3]
    rotation = Rotation.from_matrix(desired[:, :3, :3] @ current[:, :3, :3].transpose(0, 2, 1)).as_rotvec()
    values = np.c_[measured, motion, rotation]
    assert values.shape == (count, 89)
    return values
