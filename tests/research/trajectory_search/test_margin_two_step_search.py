"""Two-point margin selection must remain causal, bounded and auditable."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import guarded_two_step_search as guarded
from s2a.research.trajectory_search import margin_two_step_search as margin
from s2a.research.trajectory_search import two_step_viability as original
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_margin_objective_gradient_and_second_stage_constraints():
    values = np.r_[0.01, 0.08, -0.03, 0.02, np.linspace(-0.3, 0.4, 12)]
    jacobian = np.arange(16 * 4, dtype=float).reshape(16, 4) / 100
    before = values.copy()
    value, gradient = margin.margin_prefix_objective(values, jacobian)
    assert value > 0
    for index in range(4):
        shift = np.eye(4)[index] * 1e-7
        plus = margin.margin_prefix_objective(values + jacobian @ shift, jacobian)[0]
        minus = margin.margin_prefix_objective(values - jacobian @ shift, jacobian)[0]
        np.testing.assert_allclose((plus - minus) / 2e-7, gradient[index], atol=1e-6, rtol=0)
    hard, hard_jac, future, future_jac = margin.margin_prefix_constraints(values, jacobian, 0.4)
    np.testing.assert_allclose(hard, values[:4] - [0.05, 0.06, 0.05, 0.06], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(future, values[4:] + 0.4)
    np.testing.assert_array_equal(hard_jac, np.c_[jacobian[:4], np.zeros(4)])
    np.testing.assert_array_equal(future_jac, np.c_[jacobian[4:], np.ones(12)])
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("point,component,limit", [(0, 0, 0.0095), (1, 0, 0.0095), (0, 3, 0.047), (1, 3, 0.047)])
def test_margin_requires_each_first_two_pose_components(point, component, limit):
    errors = np.zeros((8, 6))
    errors[point, component] = limit + 0.5e-10
    assert margin.margin_prefix_satisfied(errors)
    errors[point, component] = limit + 2e-10
    assert not margin.margin_prefix_satisfied(errors)
    assert errors[point, component] < (0.01 if component == 0 else 0.05)


def test_margin_candidate_is_independently_audited_and_does_not_add_responses():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    original_namespace = original.solve_two_step_viability.__globals__
    before = {key: original_namespace[key] for key in guarded._FROZEN_GLOBALS}
    events = []
    result = margin.solve_margin_two_step(*args, on_response=events.append)
    assert result["status"] == "two_step_margin_candidate"
    assert events.count("attempted") == events.count("completed") == result["response_evaluations"] <= 320
    assert result["margin_search"]["recorded_rollout_calls"] == result["response_evaluations"]
    assert result["margin_search"]["extra_model_responses"] == 0
    assert all(original_namespace[key] is value for key, value in before.items())
    audit_args = list(audit_args)
    audit_args[11] = targets
    checked = audit_two_step_witness(result["best_joint_safe"], *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]
    assert margin.margin_prefix_satisfied(checked["tcp_errors"])


def test_nominally_admissible_but_margin_infeasible_first_pose_is_not_selected():
    args, _audit_args, targets = solve_and_audit_inputs()
    args = list(args)
    model, velocity, seed, _previous, _interface, site, _tool, _targets, _twists, lower, upper = args
    peak = seed.copy()
    peak[:, 0] = 0.25
    limit_rollout = rollout_joint_preview(model, velocity, peak, site, lower, upper, enforce_limits=False)
    targets[:] = limit_rollout["pose"]
    targets[0, 0, 3] += 0.0097
    args[2] = peak
    result = margin.solve_margin_two_step(*args)
    assert result["margin_search"]["primary_nominal_prefix_evaluations"] > 0
    assert result["status"] == "two_step_margin_no_witness"
    assert result["margin_search"]["margin_prefix_evaluations"] == 0
    assert "best_joint_safe" not in result
    assert result["response_evaluations"] <= 320


def test_margin_search_rejects_invalid_horizon_before_a_response():
    args, _audit_args, _targets = solve_and_audit_inputs()
    changed = list(args)
    changed[7] = changed[7][:7]
    events = []
    with pytest.raises(ValueError):
        margin.solve_margin_two_step(*changed, on_response=events.append)
    assert not events
