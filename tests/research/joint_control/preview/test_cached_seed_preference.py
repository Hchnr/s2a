"""The opt-in seed preference cannot bypass an existing safety or pose gate."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.cached_seed_preference import (
    CachedSeedPreferredAudit,
    choose_cached_seed,
    solve_prefer_cached_seed,
)
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import margin_guarded_fallback as shared
from s2a.research.trajectory_search import margin_two_step_search as margin
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_shared_shift_fallback import attach_completed


def candidate(value, future, *, position=0.0):
    errors = np.zeros((8, 6))
    errors[1, 0] = position
    commands = np.full((8, 6), value)
    return future, float(value), (value,), commands, None, errors, 0.0


def test_seed_is_preferred_only_when_it_is_in_the_qualified_cache():
    seed = np.ones((8, 6)) * 0.1
    best = candidate(0.0, 0.0)
    matching = candidate(0.1, 0.2)
    selected, info = choose_cached_seed([best, matching], seed)
    assert selected is matching and info["selected_changed"] is True
    assert info["future_slack_increase"] == pytest.approx(0.2)
    assert info["extra_model_responses"] == 0
    selected, info = choose_cached_seed([best], seed)
    assert selected is best and info["seed_in_eligible_cache"] is False
    with pytest.raises(ValueError, match="margins"):
        choose_cached_seed([best, candidate(0.1, 0.2, position=0.0096)], seed)


def test_seed_preference_keeps_response_budget_and_independent_witness():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    original_events, preferred_events = [], []
    original_isolator = shared.isolated_margin_solver
    original_selector = margin.select_evaluated_margin_candidate
    prior = shared.solve_margin_guarded_fallback(*args, on_response=original_events.append)
    preferred = solve_prefer_cached_seed(*args, on_response=preferred_events.append)
    assert prior["status"] == preferred["status"] == "margin_guarded_candidate"
    assert prior["response_evaluations"] == preferred["response_evaluations"] <= 320
    assert original_events == preferred_events
    assert preferred["seed_preference"]["extra_model_responses"] == 0
    assert shared.isolated_margin_solver is original_isolator
    assert margin.select_evaluated_margin_candidate is original_selector
    selected = preferred["best_joint_safe"]
    assert margin.margin_prefix_satisfied(selected["tcp_errors"])
    audit_args = list(audit_args)
    audit_args[11] = targets
    checked = audit_two_step_witness(selected, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]


def test_seed_preference_adapter_keeps_scoped_independent_audit():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    receipt, events = {}, []
    adapter = CachedSeedPreferredAudit(receipt)
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)), on_response=events.append)
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["selection_scope"] == "margin_two_step"
    assert receipt["shared_shift_traces"][0]["audit_status"] == "passed"
    assert receipt["seed_preference_traces"][0]["extra_model_responses"] == 0
    assert events.count("attempted") == events.count("completed") == witness["response_evaluations"] <= 320


def test_refresh_and_seed_preferences_cannot_be_combined():
    args, _audit_args, _targets = solve_and_audit_inputs()
    with pytest.raises(ValueError, match="cannot be combined"):
        solve_prefer_cached_seed(*args, prefer_refresh=True)
