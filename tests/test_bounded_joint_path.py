"""Actual anchoring, anticipation and invalid-seed rejection for geometric paths."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.bounded_joint_path import plan_joint_path
from s2a.full_coupled_pose_inverse import left_jacobian
from s2a.geometry import pose_error, transform


def fk(q):
    j = np.zeros((6, 7))
    j[:3, :3] = np.eye(3)
    j[3:, 3:6] = left_jacobian(q[3:6])
    return transform(q[:3], Rotation.from_rotvec(q[3:6]).as_matrix()), j


def test_actual_anchor_is_used_instead_of_smoothing_an_unreached_initial_branch():
    seed = np.zeros((4, 7))
    seed[:, 6] = -0.4
    actual = np.zeros(7)
    actual[6] = 0.3
    targets = np.tile(np.eye(4), (4, 1, 1))
    result = plan_joint_path(fk, targets, actual, seed, -np.ones(7), np.ones(7))
    np.testing.assert_array_equal(result["actual_anchor"], actual)
    assert result["energy"] < result["seed_energy"] * 1e-7
    np.testing.assert_allclose(result["joints"][:, 6], 0.3, atol=1e-5, rtol=0)


def test_future_available_targets_allow_early_motion_inside_current_pose_tube():
    seed = np.zeros((4, 7))
    seed[:, 0] = [0, 0, 0.03, 0.04]
    targets = np.asarray([fk(q)[0] for q in seed])
    result = plan_joint_path(fk, targets, np.zeros(7), seed, -np.ones(7), np.ones(7))
    assert result["energy"] < result["seed_energy"] * 0.5
    assert result["joints"][0, 0] > 0.001
    for q, target in zip(result["joints"], targets, strict=True):
        error = pose_error(target, fk(q)[0])
        assert np.linalg.norm(error[:3]) <= 0.01 + 1e-10
        assert np.linalg.norm(error[3:]) <= 0.05 + 1e-10


def test_infeasible_seed_is_rejected_without_silently_replacing_it():
    seed = np.zeros((2, 7))
    seed[:, 0] = 0.1
    with pytest.raises(ValueError, match="Verified feasible whole-path seed"):
        plan_joint_path(fk, np.tile(np.eye(4), (2, 1, 1)), np.zeros(7), seed, -np.ones(7), np.ones(7))


def test_measured_initial_penetration_is_not_clipped_out_of_the_anchor():
    actual = np.zeros(7)
    actual[6] = 1.0004
    seed = np.zeros((2, 7))
    seed[:, 6] = 1
    result = plan_joint_path(fk, np.tile(np.eye(4), (2, 1, 1)), actual, seed, -np.ones(7), np.ones(7))
    np.testing.assert_array_equal(result["actual_anchor"], actual)
    assert result["actual_initial_penetration_rad"] == pytest.approx(0.0004)
    assert np.max(result["joints"][:, 6]) <= 1.0 + 1e-12
