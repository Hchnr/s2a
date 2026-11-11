"""Separate object motion in world and in the supporting-object frame."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.geometry import transform
from s2a.research.joint_control.tracking.task_object_motion import pose_distance, task_object_motion


def test_rigidly_carried_object_moves_in_world_but_not_relative_to_bowl():
    bowl = np.eye(4)
    cheese = transform(np.array([0.03, 0, 0.04]), np.eye(3))
    motion = transform(np.array([0.01, -0.02, 0]), Rotation.from_euler("z", 0.2).as_matrix())
    result = task_object_motion(bowl, cheese, motion @ bowl, motion @ cheese)
    assert result["bowl_world"]["translation_m"] == pytest.approx(np.sqrt(0.0005))
    assert result["bowl_world"]["rotation_rad"] == pytest.approx(0.2)
    assert result["cheese_relative_to_bowl"]["translation_m"] < 1e-12
    assert result["cheese_relative_to_bowl"]["rotation_rad"] < 1e-12


def test_relative_motion_changes_when_only_cheese_moves():
    bowl, cheese = np.eye(4), np.eye(4)
    moved = transform(np.array([0, 0.015, 0]), Rotation.from_euler("x", 0.1).as_matrix())
    result = task_object_motion(bowl, cheese, bowl, moved)
    assert result["bowl_world"]["translation_m"] == 0
    assert result["cheese_relative_to_bowl"]["translation_m"] == pytest.approx(0.015)
    assert result["cheese_relative_to_bowl"]["rotation_rad"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        pose_distance(np.eye(3), moved)
