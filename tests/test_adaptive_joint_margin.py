"""Retained-margin witnesses keep original pose caps and XML bounds explicit."""

import numpy as np
import pytest

from s2a.adaptive_joint_margin import maximize_margin
from s2a.adjustable_interior_joint_plan import solve_interior_plan
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.osc import OSCInterface
from s2a.strict_adjustable_interior_joint_plan import solve_interior_plan as strict_solve


def problem(position):
    q = np.zeros(7)
    q[0] = position
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    seed = np.zeros((8, 6))
    seed[:, 3] = 1e-9
    target = rollout_joint_preview(model, np.zeros(7), seed, np.eye(4), -np.ones(7), np.ones(7), enforce_limits=False)
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
        -np.ones(7),
        np.ones(7),
    )


def test_feasible_requested_margin_and_pose_caps_are_retained():
    args = problem(0.90)
    result = maximize_margin(*args, initial_guess=args[2])
    assert result["witness_found"]
    assert result["requested_margin_satisfied"]
    np.testing.assert_allclose(result["retained_margin_rad"], 0.05, atol=1e-10, rtol=0)
    np.testing.assert_allclose(result["first_pose_caps"], [0.01, 0.05], atol=1e-12, rtol=0)
    assert result["minimum_inner_margin_rad"] >= result["retained_margin_rad"] - 1e-10
    assert np.max(abs(np.diff(np.vstack((np.zeros(6), result["planned_commands"])), axis=0))) <= 0.25 + 1e-12


@pytest.mark.parametrize("solver", [solve_interior_plan, strict_solve])
def test_smaller_exploratory_margin_does_not_change_xml_or_rotation_caps(solver):
    args = problem(0.96)
    rejected = solver(*args, joint_margin_rad=0.05)
    assert rejected["status"] == "initial_outside_requested_interior"
    accepted = solver(*args, initial_guess=args[2], joint_margin_rad=0.03)
    assert accepted["witness_found"]
    np.testing.assert_allclose(accepted["first_pose_caps"], rejected["first_pose_caps"], atol=1e-12, rtol=0)
    assert accepted["minimum_inner_margin_rad"] >= 0.03 - 1e-10
    assert accepted["joint_margin_rad"] == 0.03
    np.testing.assert_array_equal(args[-1], np.ones(7))
