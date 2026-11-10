"""Two-point feasibility must keep the prefix hard and independently audited."""

import copy

import numpy as np
import pytest

from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import staged_first_feasibility as first_only
from s2a.research.trajectory_search import two_step_viability as two
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.terminal.test_configurable_tracking_terminal_plan import inputs as tight_inputs


def test_two_point_violation_gradient_and_hard_prefix_slack():
    values = np.r_[-0.4, 0.2, -0.3, -0.1, np.ones(12)]
    jacobian = np.arange(16 * 4, dtype=float).reshape(16, 4) / 100
    value, gradient = two.prefix_violation_objective(values, jacobian)
    assert value == pytest.approx((0.4**2 + 0.3**2 + 0.1**2) / 2)
    for index in range(4):
        shift = np.eye(4)[index] * 1e-6
        plus = two.prefix_violation_objective(values + jacobian @ shift, jacobian)[0]
        minus = two.prefix_violation_objective(values - jacobian @ shift, jacobian)[0]
        np.testing.assert_allclose((plus - minus) / 2e-6, gradient[index], atol=1e-10, rtol=0)
    hard, hard_jac, future, future_jac = two.split_prefix_constraints(values, jacobian, 10.0)
    np.testing.assert_array_equal(hard, values[:4])
    np.testing.assert_array_equal(future, values[4:] + 10)
    np.testing.assert_array_equal(hard_jac[:, -1], np.zeros(4))
    np.testing.assert_array_equal(future_jac[:, -1], np.ones(12))


def test_two_point_candidate_shares_budget_and_passes_independent_audit():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.015
    targets[-1, 0, 3] += 0.2
    events = []
    result = two.solve_two_step_viability(*args, on_response=events.append)
    assert result["status"] == "two_step_candidate" and result["prefix_hard_evaluations"] > 0
    assert events.count("attempted") == events.count("completed") == result["response_evaluations"] <= 320
    assert result["initialization"]["actual_model_rollouts"] == result["response_evaluations"]
    assert sum(p["iterations"] for p in result["search_phases"] if p["iterations"] is not None) <= 120
    audit_args = list(audit_args)
    audit_args[11] = targets
    checked = audit_two_step_witness(result["best_joint_safe"], *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"]
    assert not checked["all_pose_caps_enforced"] and checked["braking_constraint_satisfied"]


def test_unreachable_second_pose_is_not_selected_as_a_witness():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 2.0
    result = two.solve_two_step_viability(*args)
    assert result["status"] == "two_step_no_witness" and "best_joint_safe" not in result
    assert result["prefix_hard_evaluations"] == 0 and result["response_evaluations"] <= 320
    assert [phase["phase"] for phase in result["search_phases"]] == ["two_point_feasibility"]
    assert result["closest_prefix_rejected"]["prefix_normalized_violation"] > 0


def test_independent_audit_rejects_a_first_only_candidate_with_bad_second_pose():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 1.0
    result = first_only.solve_staged_first_feasibility(*args)
    candidate = copy.deepcopy(result["best_joint_safe"])
    candidate.update(hard_pose_prefix=2, prefix_selection_hard=True)
    audit_args = list(audit_args)
    audit_args[11] = targets
    with pytest.raises(ValueError, match="Both first pose points"):
        audit_two_step_witness(candidate, *audit_args, past_inner_error_rad=np.zeros((25, 7)))


def test_tight_joint_seed_records_original_feasible_initializer():
    result = two.solve_two_step_viability(*tight_inputs())
    assert result["initialization"]["warm_start_used"]
    assert result["initialization"]["check"]["accepted"]
    assert 1 <= result["initialization"]["start_lp_attempts"] <= 2
    assert result["response_evaluations"] == result["initialization"]["actual_model_rollouts"] <= 320
