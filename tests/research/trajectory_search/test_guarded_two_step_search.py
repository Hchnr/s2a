"""Numerical guarding must leave original acceptance and audit unchanged."""

import numpy as np

from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import guarded_two_step_search as guarded
from s2a.research.trajectory_search import two_step_viability as original
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_scaled_guarded_gradient_matches_finite_differences():
    values = np.r_[-0.04, 0.02, -0.03, 0.00005, np.ones(12)]
    jacobian = np.arange(16 * 4, dtype=float).reshape(16, 4) / 100
    value, gradient = guarded.guarded_prefix_objective(values, jacobian)
    assert value > 0
    for index in range(4):
        shift = np.eye(4)[index] * 1e-7
        plus = guarded.guarded_prefix_objective(values + jacobian @ shift, jacobian)[0]
        minus = guarded.guarded_prefix_objective(values - jacobian @ shift, jacobian)[0]
        np.testing.assert_allclose((plus - minus) / 2e-7, gradient[index], atol=1e-6, rtol=0)
    inside = np.r_[np.full(4, 2e-4), np.ones(12)]
    assert guarded.guarded_prefix_objective(inside, jacobian)[0] == 0


def test_isolated_solver_does_not_mutate_original_namespace():
    candidate = guarded.isolated_solver()
    assert candidate.__code__ is original.solve_two_step_viability.__code__
    assert candidate.__globals__ is not original.solve_two_step_viability.__globals__
    assert candidate.__globals__["prefix_violation_objective"] is guarded.guarded_prefix_objective
    assert (
        original.solve_two_step_viability.__globals__["prefix_violation_objective"]
        is original.prefix_violation_objective
    )


def test_guarded_candidate_keeps_original_audit_and_shared_response_budget():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.015
    targets[-1, 0, 3] += 0.2
    events = []
    result = guarded.solve_guarded_two_step(*args, on_response=events.append)
    assert result["status"] == "two_step_candidate"
    assert events.count("attempted") == events.count("completed") == result["response_evaluations"] <= 320
    assert result["first_stage_search_objective"]["acceptance_caps_changed"] is False
    audit_args = list(audit_args)
    audit_args[11] = targets
    checked = audit_two_step_witness(result["best_joint_safe"], *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]


def test_guarded_search_does_not_select_unreachable_second_pose():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 2.0
    result = guarded.solve_guarded_two_step(*args)
    assert result["status"] == "two_step_no_witness" and "best_joint_safe" not in result
    assert result["response_evaluations"] <= 320
