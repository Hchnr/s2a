"""Accepted Cartesian states must be FK of the accepted predicted joint states."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_configurable_tracking_terminal_plan import inputs

from s2a.fk_consistent_joint_preview import fk_consistent_rollout
from s2a.fk_consistent_trajectory_joint_plan import solve_interior_plan
from s2a.full_coupled_pose_inverse import left_jacobian
from s2a.geometry import pose_error, transform
from s2a.interior_joint_preview import rollout_joint_preview


def setup():
    args = list(inputs())
    anchor = args[0]["q"].copy()

    def fk(q):
        change = q - anchor
        jac = np.zeros((6, 7))
        jac[:3, :3] = np.eye(3)
        jac[3:, 3:6] = left_jacobian(change[3:6])
        return transform(change[:3], Rotation.from_rotvec(change[3:6]).as_matrix()), jac

    args[2] = np.zeros((8, 6))
    args[2][:, 3] = 1e-9
    raw = rollout_joint_preview(args[0], args[1], args[2], args[5], args[9], args[10], enforce_limits=False)
    consistent = fk_consistent_rollout(raw, fk)
    args[7], args[8] = consistent["pose"], consistent["twist"]
    return args, fk


def test_witness_has_exact_fk_endpoints_and_retains_original_joint_constraints():
    args, fk = setup()
    result = solve_interior_plan(*args, fk_jacobian=fk, initial_guess=args[2])
    assert result["witness_found"] and result["all_pose_caps_enforced"]
    assert result["response_evaluations"] <= 160 and result["iterations"] <= 60
    commands = result["planned_commands"]
    raw = rollout_joint_preview(
        args[0],
        args[1],
        np.asarray([args[4].decode(c) for c in commands]),
        args[5],
        args[9],
        args[10],
        enforce_limits=False,
    )
    actual = fk_consistent_rollout(raw, fk)
    np.testing.assert_allclose(actual["pose"], result["rollout"]["pose"], atol=1e-12, rtol=0)
    for i, q in enumerate(actual["joint_position"]):
        pose, jac = fk(q)
        np.testing.assert_array_equal(pose, result["rollout"]["pose"][i])
        np.testing.assert_allclose(jac @ actual["joint_velocity"][i], result["rollout"]["twist"][i], atol=1e-12)
        error = pose_error(args[7][i], pose)
        assert np.linalg.norm(error[:3]) <= 0.01 + 1e-10
        assert np.linalg.norm(error[3:]) <= 0.05 + 1e-10
    q, margin = actual["inner_joint_position"], result["requested_position_margin_rad"]
    assert np.min(np.minimum(q - args[9] - margin, args[10] - margin - q)) >= -1e-10
    braking = actual["joint_position"] + 0.1 * actual["joint_velocity"]
    assert np.min(np.minimum(braking - args[9], args[10] - braking)) >= 0.05 - 1e-10
    assert np.all(result["terminal_normals"] @ actual["joint_velocity"][-1] <= 1e-10)
    assert np.max(abs(commands)) <= 1 + 1e-12
    assert np.max(abs(np.diff(np.vstack((args[3], commands)), axis=0))) <= 0.25 + 1e-12


def test_fk_frame_mismatch_is_rejected_before_search():
    args, fk = setup()
    args[5] = args[5].copy()
    args[5][0, 3] += 0.1
    with pytest.raises(AssertionError):
        solve_interior_plan(*args, fk_jacobian=fk)
