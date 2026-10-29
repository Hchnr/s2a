"""Positive buffer changes preserve true bounds, target data and normal behavior."""

import numpy as np

from s2a.configurable_tracking_terminal_plan import solve_interior_plan
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface
from s2a.tracking_terminal_joint_plan import solve_interior_plan as original_solve


def inputs():
    q = np.zeros(7)
    q[0] = 0.94
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.full(6, 100.0), np.full(6, 20.0), q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 0] = 0.25
    seed[:, 3] = 1e-9
    lo, hi = -np.ones(7), np.ones(7)
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), lo, hi, enforce_limits=False)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    return (
        model,
        np.zeros(7),
        seed,
        np.zeros(6),
        interface,
        np.eye(4),
        np.eye(4),
        target["pose"],
        target["twist"],
        lo,
        hi,
    )


def test_default_exactly_preserves_original_commands_and_objective():
    args = inputs()
    old = original_solve(*args)
    new = solve_interior_plan(*args)
    np.testing.assert_array_equal(new["planned_commands"], old["planned_commands"])
    assert new["objective"] == old["objective"]
    assert new["joint_margin_rad"] == 0.05


def test_smaller_positive_buffer_keeps_true_limits_and_improves_tracking_model():
    args = inputs()
    targets = args[7].copy()
    original = solve_interior_plan(*args)
    changed = solve_interior_plan(*args, joint_margin_rad=0.01)
    assert changed["witness_found"]
    assert changed["objective"] < original["objective"]
    assert changed["minimum_inner_margin_rad"] >= 0.01 - 1e-10
    assert np.all(changed["terminal_signed_velocities_rad_s"] <= 1e-10)
    np.testing.assert_array_equal(args[7], targets)
    np.testing.assert_array_equal(args[-1], np.ones(7))
    plan = changed["planned_commands"]
    assert np.max(abs(plan)) <= 1 + 1e-12
    assert np.max(abs(np.diff(np.vstack((args[3], plan)), axis=0))) <= 0.25 + 1e-12
