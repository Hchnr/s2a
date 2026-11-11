"""Shared search budgets and reused starts must fail closed."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import feasible_start_restoration as feasible
from s2a.research.trajectory_search import guarded_two_step_search as guarded
from s2a.research.trajectory_search import margin_guarded_fallback as policy
from s2a.research.trajectory_search import margin_two_step_search as margin
from s2a.research.trajectory_search import two_step_viability as original
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_margin_candidate_has_priority_and_preserves_original_solver_globals():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    before = {key: original.solve_two_step_viability.__globals__[key] for key in guarded._FROZEN_GLOBALS}
    events = []
    result = policy.solve_margin_guarded_fallback(*args, on_response=events.append)
    trace = result["shared_search"]
    assert result["status"] == "margin_guarded_candidate" and trace["selected_scope"] == "margin_two_step"
    assert trace["fallback_status"] == "not_attempted" and trace["fallback_responses"] == 0
    assert events.count("attempted") == events.count("completed") == result["response_evaluations"] <= 320
    assert all(original.solve_two_step_viability.__globals__[key] is value for key, value in before.items())
    audit_args = list(audit_args)
    audit_args[11] = targets
    verified = audit_two_step_witness(result["best_joint_safe"], *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert verified["passed"] and margin.margin_prefix_satisfied(verified["tcp_errors"])


def test_guarded_fallback_reuses_start_and_passes_original_independent_audit(monkeypatch):
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.015
    targets[-1, 0, 3] += 0.2

    def forbidden_lp(*_args, **_kwargs):
        raise AssertionError("The already-safe seed must not call an LP")

    monkeypatch.setattr(feasible, "linprog", forbidden_lp)
    events = []
    result = policy.solve_margin_guarded_fallback(*args, on_response=events.append)
    trace = result["shared_search"]
    assert result["status"] == "margin_guarded_candidate" and trace["selected_scope"] == "guarded_two_step"
    assert trace["fallback_initializer_reused"] and trace["fallback_lp_attempts"] == 0
    assert trace["primary_responses"] + trace["fallback_responses"] == result["response_evaluations"] <= 320
    assert events.count("attempted") == events.count("completed") == result["response_evaluations"]
    first = trace["primary_feasibility_iterations"] + trace["optimizer_calls"][0]["iterations"]
    total = trace["primary_total_iterations"] + sum(call["iterations"] for call in trace["optimizer_calls"])
    assert first <= 60 and total <= 120
    audit_args = list(audit_args)
    audit_args[11] = targets
    verified = audit_two_step_witness(result["best_joint_safe"], *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert verified["passed"] and verified["prefix_pose_caps_satisfied"] and verified["braking_constraint_satisfied"]


def test_fallback_stops_at_shared_response_limit_without_dispatch(monkeypatch):
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.015
    targets[-1, 0, 3] += 0.2
    monkeypatch.setattr(policy, "MAX_RESPONSES", 3)
    events = []
    result = policy.solve_margin_guarded_fallback(*args, on_response=events.append)
    assert result["status"] == "margin_guarded_no_witness" and "best_joint_safe" not in result
    assert result["shared_search"]["fallback_status"] == "SharedResponseBudgetExhausted"
    assert events.count("attempted") == events.count("completed") == result["response_evaluations"] == 3
    assert result["shared_search"]["optimizer_calls"][-1]["iterations"] is None


def test_unavailable_primary_iterations_cannot_open_fallback(monkeypatch):
    args, _audit_args, _targets = solve_and_audit_inputs()

    def fake_isolated(_capture):
        def fake_solver(*_args, on_response, **_kwargs):
            on_response("attempted")
            on_response("completed")
            return dict(
                status="two_step_margin_no_witness",
                response_evaluations=1,
                search_phases=[dict(phase="two_point_feasibility", iterations=None)],
                initialization=dict(start_lp_attempts=0),
            )

        return fake_solver

    monkeypatch.setattr(policy, "isolated_margin_solver", fake_isolated)
    result = policy.solve_margin_guarded_fallback(*args)
    assert result["status"] == "margin_guarded_no_witness"
    assert result["shared_search"]["fallback_status"] == "primary_iteration_count_unknown"
    assert result["shared_search"]["fallback_responses"] == 0


def test_fallback_optimizers_share_feasibility_and_total_iterations():
    calls = []

    def minimize(*_args, **kwargs):
        calls.append(kwargs["options"]["maxiter"])
        return SimpleNamespace(nit=1)

    receipt = {}
    bounded = policy.bounded_minimize(minimize, primary_first=58, primary_total=119, receipt=receipt)
    bounded(None, None, options={"maxiter": 60})
    assert calls == [1] and receipt["optimizer_calls"][0]["iterations"] == 1
    with pytest.raises(policy.SharedIterationBudgetExhausted):
        bounded(None, None, options={"maxiter": 120})


def test_reused_initializer_rejects_changed_domain_or_seed():
    args, _audit_args, _targets = solve_and_audit_inputs()
    model, velocity, seed, previous, interface, site, _tool, _targets, _twists, lower, upper = args
    rollout = rollout_joint_preview(model, velocity, seed, site, lower, upper, enforce_limits=False)
    domain = feasible.linear_domain(model, seed, previous, interface, lower, upper, np.zeros((25, 7)), rollout)
    info = dict(seed_linear_safe=True, warm_start_used=False, start_lp_attempts=0)
    initialize = policy.reusable_initializer(seed, info, domain, feasible.domain_check)
    start, reused = initialize(seed, domain, interface)
    np.testing.assert_array_equal(start, seed)
    assert reused["primary_start_reused"] and reused["start_lp_attempts"] == 0
    changed = {key: value.copy() for key, value in domain.items()}
    changed["joint_hi"][0] += 1e-6
    with pytest.raises(ValueError, match="exactly match the primary domain"):
        initialize(seed, changed, interface)
    changed_seed = seed.copy()
    changed_seed[0, 0] += 1e-6
    with pytest.raises(ValueError, match="seed must match"):
        initialize(changed_seed, domain, interface)
