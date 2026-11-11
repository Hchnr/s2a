"""Compare task-object rigid poses in world and in the supporting-object frame."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import inverse, validate_transform


def _one_pose(value):
    pose = validate_transform(np.asarray(value, dtype=float))
    if pose.shape != (4, 4):
        raise ValueError("One rigid object pose is required")
    return pose


def pose_distance(first, second):
    """Return translational and SO(3) geodesic displacement between poses."""
    first, second = _one_pose(first), _one_pose(second)
    return {
        "translation_m": float(np.linalg.norm(second[:3, 3] - first[:3, 3])),
        "rotation_rad": float(Rotation.from_matrix(first[:3, :3].T @ second[:3, :3]).magnitude()),
    }


def task_object_motion(initial_bowl, initial_cheese, current_bowl, current_cheese):
    """Measure bowl/cheese world motion and cheese motion in the bowl frame."""
    b0, c0, b1, c1 = (_one_pose(pose) for pose in (initial_bowl, initial_cheese, current_bowl, current_cheese))
    return {
        "bowl_world": pose_distance(b0, b1),
        "cheese_world": pose_distance(c0, c1),
        "cheese_relative_to_bowl": pose_distance(inverse(b0) @ c0, inverse(b1) @ c1),
    }
