"""World twist from available generated endpoints and the measured query anchor."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import validate_transform
from s2a.reference_linear_velocity import reference_linear_velocity


def reference_world_twist(poses, current_index, anchor, dt):
    targets, anchor = validate_transform(np.asarray(poses)), validate_transform(np.asarray(anchor))
    linear = reference_linear_velocity(targets, current_index, anchor, dt)
    before = targets[current_index - 1] if current_index else anchor
    after = targets[linear["after_index"]]
    interval = dt * (2 if linear["has_next_generated_reference"] else 1)
    rotation_delta = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec()
    if np.linalg.norm(rotation_delta) >= np.pi - 1e-6:
        raise ValueError("Generated angular difference at pi is ambiguous")
    return dict(**linear, twist=np.r_[linear["velocity"], rotation_delta / interval], angular_frame="MuJoCo world")
