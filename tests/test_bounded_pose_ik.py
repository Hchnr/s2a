"""Constructive full-pose IK and honest residuals when bounds prevent a witness."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.bounded_pose_ik import solve_pose_ik
from s2a.full_coupled_pose_inverse import left_jacobian
from s2a.geometry import pose_error, transform


def fk(q):
    pose = transform(q[:3], Rotation.from_rotvec(q[3:6]).as_matrix())
    jacobian = np.zeros((6, 7))
    jacobian[:3, :3] = np.eye(3)
    jacobian[3:, 3:6] = left_jacobian(q[3:6])
    return pose, jacobian


def test_constructs_nontrivial_so3_pose_with_redundant_joint():
    desired, _ = fk(np.array([0.1, 0.05, -0.02, 0.2, -0.3, 0.1, 0.4]))
    result = solve_pose_ik(fk, desired, np.zeros(7), -np.ones(7), np.ones(7))
    assert result["geometric_witness"] is True
    np.testing.assert_allclose(pose_error(desired, fk(result["joints"])[0]), np.zeros(6), atol=1e-7, rtol=0)


def test_bounded_minimum_is_not_mislabeled_as_a_pose_witness():
    desired = np.eye(4)
    desired[0, 3] = 2.0
    result = solve_pose_ik(fk, desired, np.zeros(7), -np.ones(7) * 0.1, np.ones(7) * 0.1)
    assert result["in_bounds"] is True
    assert result["geometric_witness"] is False
    assert np.linalg.norm(result["error"][:3]) > 1.8


def test_solver_start_clipping_is_explicit():
    initial = np.zeros(7)
    initial[6] = 1.0004
    result = solve_pose_ik(fk, np.eye(4), initial, -np.ones(7), np.ones(7))
    assert abs(result["initial_clip_max_abs_rad"] - 0.0004) < 1e-12
    assert result["geometric_witness"] is True
