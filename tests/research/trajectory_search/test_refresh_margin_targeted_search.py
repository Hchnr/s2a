"""Check the added margin and future-quality gates independently of SLSQP."""

import numpy as np
import pytest

from s2a.research.trajectory_search.margin_two_step_search import solve_margin_two_step
from s2a.research.trajectory_search.refresh_margin_targeted_search import (
    candidate_is_eligible,
    solve_targeted_refresh,
    targeted_constraints,
)
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_targeted_constraints_only_tighten_second_pose_and_bound_the_suffix():
    values = np.arange(16, dtype=float) / 100
    jacobian = np.arange(16 * 3, dtype=float).reshape(16, 3)
    adjusted, actual_jacobian = targeted_constraints(values, jacobian, 0.08)
    np.testing.assert_allclose(adjusted[:4], values[:4] - [0.05, 0.06, 0.075, 0.09])
    np.testing.assert_allclose(adjusted[4:], values[4:] + 0.10)
    np.testing.assert_array_equal(actual_jacobian, jacobian)


@pytest.mark.parametrize("failure", [None, "first", "second_position", "second_rotation", "future", "safety"])
def test_candidate_must_pass_every_original_and_added_gate(failure):
    errors = np.zeros((8, 6))
    errors[0, 0], errors[0, 3] = 0.0094, 0.046
    errors[1, 0], errors[1, 3] = 0.0092, 0.0454
    future, safety = 0.099, {"accepted": True}
    if failure == "first":
        errors[0, 0] = 0.0096
    elif failure == "second_position":
        errors[1, 0] = 0.0093
    elif failure == "second_rotation":
        errors[1, 3] = 0.0456
    elif failure == "future":
        future = 0.101
    elif failure == "safety":
        safety["accepted"] = False
    assert candidate_is_eligible(errors, future, 0.08, safety) is (failure is None)


def test_constraints_reject_a_negative_or_nonfinite_baseline():
    for baseline in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            targeted_constraints(np.zeros(16), np.zeros((16, 48)), baseline)


def test_targeted_search_uses_only_evaluated_candidates_and_counts_extra_responses():
    args, _audit_args, _targets = solve_and_audit_inputs()
    baseline = solve_margin_two_step(*args)["best_joint_safe"]
    targeted_args = list(args)
    targeted_args[2] = baseline["planned_commands"]
    events = []
    result = solve_targeted_refresh(
        *targeted_args,
        past_inner_error_rad=np.zeros((25, 7)),
        expected_seed_tcp_errors=baseline["tcp_errors"],
        expected_baseline_future_slack=baseline["future_quality_slack"],
        on_response=events.append,
    )
    assert result["response_attempts"] == result["response_completions"] == len(events) // 2 <= 320
    assert events[::2] == ["attempted"] * (len(events) // 2)
    assert events[1::2] == ["completed"] * (len(events) // 2)
    assert result["eligible_candidates"] > 0
    assert candidate_is_eligible(
        result["candidate"]["tcp_errors"],
        result["candidate"]["future_quality_slack"],
        result["baseline_future_slack"],
        result["candidate"]["linear_safety"],
    )
