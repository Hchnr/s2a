"""Margin penalty derivatives and permitted first-pose preventive adjustment."""

import numpy as np

from s2a.fast_long_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.geometry import pose_error
from s2a.osc import OSCInterface
from s2a.preventive_joint_inverse import margin_cost, refine_long_plan


def test_margin_hinge_gradient_matches_independent_centered_difference():
    rng = np.random.default_rng(42)
    q = rng.uniform(-1.1, 1.1, (8, 7))
    derivative = rng.normal(size=(8, 7, 12))
    rollout = dict(joint_position=q, joint_position_jacobian=derivative)
    _, gradient = margin_cost(rollout, -np.ones(7), np.ones(7))
    numeric = []
    for k in range(12):
        plus = dict(joint_position=q + 1e-7 * derivative[:, :, k], joint_position_jacobian=derivative)
        minus = dict(joint_position=q - 1e-7 * derivative[:, :, k], joint_position_jacobian=derivative)
        numeric.append(
            (margin_cost(plus, -np.ones(7), np.ones(7))[0] - margin_cost(minus, -np.ones(7), np.ones(7))[0]) / 2e-7
        )
    np.testing.assert_allclose(gradient, numeric, atol=1e-6, rtol=1e-6)


def test_prevention_reduces_margin_cost_while_first_pose_remains_in_declared_tube():
    q = np.zeros(7)
    q[0] = 0.98
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 3] = 1e-9
    lo, hi = -np.ones(7), np.ones(7)
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), lo, hi, enforce_limits=True)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    result = refine_long_plan(
        model, np.zeros(7), seed, np.zeros(6), interface, np.eye(4), np.eye(4), target["pose"], target["twist"], lo, hi
    )
    assert result["accepted"]
    assert result["objective"] < result["seed_objective"]
    assert result["margin_objective"] < result["seed_margin_objective"]
    error = pose_error(target["pose"][0], result["rollout"]["pose"][0])
    assert 1e-6 < np.linalg.norm(error[:3]) <= 0.01 + 1e-10
    assert np.linalg.norm(error[3:]) <= 0.05 + 1e-10
    assert np.max(abs(np.diff(np.vstack((np.zeros(6), result["planned_commands"])), axis=0))) <= 0.25 + 1e-12
