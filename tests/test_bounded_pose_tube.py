"""Pose tube preserves admissible configurations and avoids exact-IK motion."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.bounded_pose_tube import closest_pose_in_tube
from s2a.full_coupled_pose_inverse import left_jacobian
from s2a.geometry import transform


def fk(q):
    jacobian = np.zeros((6, 7))
    jacobian[:3, :3] = np.eye(3)
    jacobian[3:, 3:6] = left_jacobian(q[3:6])
    return transform(q[:3], Rotation.from_rotvec(q[3:6]).as_matrix()), jacobian


def test_inside_tube_preserves_previous_joints_exactly():
    previous = np.array([0.004, 0, 0, 0.02, 0, 0, 0.4])
    result = closest_pose_in_tube(fk, np.eye(4), previous, -np.ones(7), np.ones(7))
    np.testing.assert_array_equal(result["joints"], previous)
    assert result["within_tube"] is True
    assert result["cold_ik_nfev"] == result["optimizer_iterations"] == 0


def test_nearest_tube_point_stops_at_tolerance_instead_of_exact_pose():
    desired = np.eye(4)
    desired[0, 3] = 0.05
    result = closest_pose_in_tube(fk, desired, np.zeros(7), -np.ones(7), np.ones(7))
    assert result["within_tube"] is True
    assert abs(result["joints"][0] - 0.04) < 1e-6
    assert np.linalg.norm(result["error"][:3]) <= 0.01 + 1e-10
    assert np.linalg.norm(result["error"][3:]) <= 0.05 + 1e-10


def test_no_tube_witness_is_not_mislabeled():
    desired = np.eye(4)
    desired[0, 3] = 2
    result = closest_pose_in_tube(fk, desired, np.zeros(7), -np.ones(7) * 0.1, np.ones(7) * 0.1)
    assert result["within_tube"] is False
    assert np.linalg.norm(result["error"][:3]) > 1.8
