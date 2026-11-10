"""Only completed periods affect tightening; witness checks are independent."""

import numpy as np
import pytest

from s2a.geometry import pose_error
from s2a.research.joint_control.tracking.early_quality_diagnostic import (
    completed_error_profile,
    independently_check_witness,
    state_before_period,
)
from s2a.research.trajectory_search.past_inner_error_joint_plan import solve_interior_plan as joint_solve
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan
from tests.research.joint_control.terminal.test_configurable_tracking_terminal_plan import inputs


def test_profile_uses_only_strictly_completed_periods_and_inner_time_prefixes():
    periods = [
        {"step": 10 + i, "actual_inner": [{"q": np.full(7, i + k / 100.0)} for k in range(25)]} for i in range(3)
    ]
    asked = []

    def forecast(i):
        asked.append(i)
        return np.zeros((25, 7))

    result = completed_error_profile(periods, 10, 2, forecast)
    assert asked == [0, 1]
    assert result["completed_periods"] == 2 and result["future_guarantee"] is False
    np.testing.assert_allclose(result["inner_position_error_rad"][:, 0], 1 + np.arange(25) / 100.0)
    periods[2]["actual_inner"][0]["q"][:] = 1000
    np.testing.assert_array_equal(
        completed_error_profile(periods, 10, 2, forecast)["inner_position_error_rad"],
        result["inner_position_error_rad"],
    )


def test_profile_rejects_missing_completed_period():
    with pytest.raises(ValueError, match="Consecutive"):
        completed_error_profile([{"step": 2, "actual_inner": []}], 1, 1, lambda _i: np.zeros((25, 7)))


def test_current_snapshot_never_reads_its_own_or_later_actual_state():
    plan = np.zeros((8, 6))
    periods = [
        {"step": 10, "actual_state": np.ones(79), "command": np.ones(7) * 0.1, "decision": {"refined_plan": plan}},
        {"step": 11, "decision": {"refined_plan": plan}},
        {"step": 12},
    ]
    state, previous, actual_plan = state_before_period(np.zeros(79), np.zeros(7), periods, 10, 1)
    np.testing.assert_array_equal(state, np.ones(79))
    np.testing.assert_array_equal(previous, np.ones(7) * 0.1)
    np.testing.assert_array_equal(actual_plan, plan)


def test_independent_rollout_checks_model_witness_and_rejects_tampering():
    args = list(inputs())
    profile = np.full((25, 7), 0.003)
    start = joint_solve(*args, past_inner_error_rad=profile)
    args[7], args[8] = start["rollout"]["pose"].copy(), start["rollout"]["twist"].copy()
    witness = solve_interior_plan(*args, initial_guess=start["planned_commands"], past_inner_error_rad=profile)
    assert witness["witness_found"]
    model, velocity, _seed, previous, interface, site, tool, targets, _twists, lower, upper = args
    check = independently_check_witness(
        witness,
        model["mass"],
        model["jacobian"],
        np.full(6, 100.0),
        np.full(6, 20.0),
        model["q"],
        velocity,
        model["q"],
        interface,
        previous,
        site,
        tool,
        targets[0],
        lower,
        upper,
        past_inner_error_rad=profile,
    )
    assert check["passed"] and check["independent_endpoint_queries"] == 200
    witness["rollout"]["inner_joint_position"][0, 0] += 0.01
    with pytest.raises(ValueError, match="differs"):
        independently_check_witness(
            witness,
            model["mass"],
            model["jacobian"],
            np.full(6, 100.0),
            np.full(6, 20.0),
            model["q"],
            velocity,
            model["q"],
            interface,
            previous,
            site,
            tool,
            targets[0],
            lower,
            upper,
            past_inner_error_rad=profile,
        )


def test_joint_only_witness_reports_but_does_not_claim_first_pose_quality():
    args = list(inputs())
    profile = np.full((25, 7), 0.003)
    witness = joint_solve(*args, past_inner_error_rad=profile)
    assert witness["witness_found"]
    args[7] = args[7].copy()
    args[7][:, 0, 3] += 1.0
    witness["first_tcp_error"] = pose_error(args[7][0], witness["rollout"]["pose"][0] @ args[6])
    model, velocity, _seed, previous, interface, site, tool, targets, _twists, lower, upper = args
    audit_args = (
        witness,
        model["mass"],
        model["jacobian"],
        np.full(6, 100.0),
        np.full(6, 20.0),
        model["q"],
        velocity,
        model["q"],
        interface,
        previous,
        site,
        tool,
        targets[0],
        lower,
        upper,
    )
    checked = independently_check_witness(*audit_args, past_inner_error_rad=profile, enforce_pose_caps=False)
    assert checked["passed"] is True
    assert checked["first_pose_caps_enforced"] is False and checked["first_pose_caps_satisfied"] is False
    with pytest.raises(ValueError, match="first-pose caps"):
        independently_check_witness(*audit_args, past_inner_error_rad=profile)
