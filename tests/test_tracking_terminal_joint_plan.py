"""Whole-plan tracking may exceed the reported local cap without changing targets."""

import numpy as np

from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface
from s2a.tracking_terminal_joint_plan import solve_interior_plan


def test_original_reference_and_bounds_are_kept_when_first_cap_is_not_enforced():
    q = np.zeros(7)
    q[0] = 0.94
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 0] = 0.25
    seed[:, 3] = 1e-9
    lo, hi = -np.ones(7), np.ones(7)
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), lo, hi, enforce_limits=False)
    original = target["pose"].copy()
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    result = solve_interior_plan(
        model, np.zeros(7), seed, np.zeros(6), interface, np.eye(4), np.eye(4), target["pose"], target["twist"], lo, hi
    )
    assert result["witness_found"]
    assert result["first_pose_caps_enforced"] is False
    assert result["original_first_pose_caps_satisfied"] is False
    assert np.linalg.norm(result["first_tcp_error"][:3]) > 0.01
    assert result["minimum_inner_margin_rad"] >= 0.05 - 1e-10
    assert np.all(result["terminal_signed_velocities_rad_s"] <= 1e-10)
    np.testing.assert_array_equal(target["pose"], original)
    np.testing.assert_array_equal(hi, np.ones(7))
    assert np.max(abs(np.diff(np.vstack((np.zeros(6), result["planned_commands"])), axis=0))) <= 0.25 + 1e-12
