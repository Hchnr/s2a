"""Multi-period precision constraints and independent reconstruction."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.controllers.osc import OSCInterface
from s2a.research.joint_control.limits.joint_limit_preview_objective import preview_objective
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.full_horizon_quality import pose_quality_constraints, solve_full_horizon


def fixture():
    q, velocity = np.zeros(7), np.zeros(7)
    mass, jacobian, kp, kd = np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.full(6, 100.0), np.full(6, 20.0)
    model = build_joint_preview(mass, jacobian, kp, kd, q, q, null_kp=10, dt=0.002)
    seed = np.tile([0.01, 0, 0, 0.001, 0, 0], (8, 1))
    lower, upper = -np.ones(7), np.ones(7)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    rollout = rollout_joint_preview(model, velocity, seed, np.eye(4), lower, upper, enforce_limits=False)
    args = (
        model,
        velocity,
        seed,
        np.zeros(6),
        interface,
        np.eye(4),
        np.eye(4),
        rollout["pose"].copy(),
        rollout["twist"].copy(),
        lower,
        upper,
    )
    return args, (mass, jacobian, kp, kd, q, velocity)


def test_multi_period_constraint_jacobian_matches_finite_differences():
    args, _ = fixture()
    model, velocity, seed, _, interface, site, tool, target, twists, lower, upper = args
    target[:, :3, 3] += np.array([0.001, -0.002, 0.0005])
    target[:, :3, :3] = Rotation.from_rotvec([0.01, -0.004, 0.003]).as_matrix() @ target[:, :3, :3]

    def evaluate(x):
        commands = x.reshape(8, 6)
        rollout = rollout_joint_preview(
            model, velocity, [interface.decode(row) for row in commands], site, lower, upper, enforce_limits=False
        )
        return pose_quality_constraints(preview_objective(rollout, tool, target, twists), 8, np.ones(48))

    x = seed.ravel().copy()
    values, analytic, _errors = evaluate(x)
    for column in (0, 3, 15, 29, 47):
        plus, minus = x.copy(), x.copy()
        plus[column] += 1e-6
        minus[column] -= 1e-6
        numerical = (evaluate(plus)[0] - evaluate(minus)[0]) / 2e-6
        np.testing.assert_allclose(analytic[:, column], numerical, atol=2e-5, rtol=2e-4)
    assert values.shape == (16,) and analytic.shape == (16, 48)


def test_later_bad_reference_is_detected_when_first_pose_is_good():
    args, _ = fixture()
    model, velocity, seed, _, _interface, site, tool, targets, twists, lower, upper = args
    rollout = rollout_joint_preview(model, velocity, seed, site, lower, upper, enforce_limits=False)
    targets[-1, 0, 3] += 0.1
    values, _, _ = pose_quality_constraints(preview_objective(rollout, tool, targets, twists), 8, np.ones(48))
    assert values[0] > 0 and values[-2] < 0


def test_witness_reconstructs_all_tcp_points_and_rejects_tampering():
    args, source = fixture()
    result = solve_full_horizon(*args)
    assert result["witness_found"] and result["pose_horizon"] == 8
    model, velocity, _seed, previous, interface, site, tool, targets, _twists, lower, upper = args
    mass, jacobian, kp, kd, q, _ = source
    check_args = (
        result,
        mass,
        jacobian,
        kp,
        kd,
        q,
        velocity,
        q,
        interface,
        previous,
        site,
        tool,
        targets,
        lower,
        upper,
    )
    checked = audit_full_horizon_witness(*check_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["all_pose_caps_satisfied"] and checked["pose_horizon"] == 8
    damaged = copy.deepcopy(result)
    damaged["tcp_errors"][4, 0] += 0.01
    with pytest.raises(AssertionError):
        audit_full_horizon_witness(damaged, *check_args[1:], past_inner_error_rad=np.zeros((25, 7)))
