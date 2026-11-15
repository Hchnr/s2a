"""The bounded preference stays inside the existing search and witness gates."""

import numpy as np
import pytest

from s2a.research.joint_control.preview import cached_seed_preference as unbounded
from s2a.research.joint_control.preview.bounded_seed_preference import (
    MAX_FUTURE_SLACK_INCREASE,
    BoundedSeedPreferredAudit,
    choose_bounded_seed,
    solve_bounded_seed,
)
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import margin_guarded_fallback as shared
from s2a.research.trajectory_search import margin_two_step_search as margin
from tests.research.joint_control.preview.test_cached_seed_preference import candidate
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_shared_shift_fallback import attach_completed


def test_bounded_choice_preserves_default_when_seed_exceeds_the_limit():
    seed = np.full((8, 6), 0.1)
    default, matching = candidate(0.0, 0.1), candidate(0.1, 0.41)
    selected, trace = choose_bounded_seed([default, matching], seed)
    assert selected is default
    assert trace["seed_in_eligible_cache"] is True
    assert trace["seed_blocked_by_future_slack"] is True
    assert trace["selected_seed"] is trace["selected_changed"] is False
    assert trace["unbounded_seed_future_slack_increase"] == pytest.approx(0.31)
    assert trace["future_slack_increase"] == 0 and trace["extra_model_responses"] == 0


def test_bounded_choice_accepts_boundary_and_falls_back_without_seed():
    seed = np.full((8, 6), 0.1)
    default = candidate(0.0, 0.1)
    boundary = candidate(0.1, 0.1 + MAX_FUTURE_SLACK_INCREASE)
    selected, trace = choose_bounded_seed([default, boundary], seed)
    assert selected is boundary and trace["selected_changed"] is True
    assert trace["seed_blocked_by_future_slack"] is False
    selected, trace = choose_bounded_seed([default], seed)
    assert selected is default and trace["seed_in_eligible_cache"] is False
    assert trace["unbounded_seed_future_slack_increase"] is None
    with pytest.raises(ValueError, match="margins"):
        choose_bounded_seed([default, candidate(0.1, 0.2, position=0.0096)], seed)
    with pytest.raises(ValueError, match="nonempty"):
        choose_bounded_seed([], seed)


def test_bounded_solver_keeps_the_existing_response_and_witness_contract():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    baseline_events, bounded_events = [], []
    original_selector = unbounded.choose_cached_seed
    original_isolator = shared.isolated_margin_solver
    original_ranking = margin.select_evaluated_margin_candidate
    baseline = shared.solve_margin_guarded_fallback(*args, on_response=baseline_events.append)
    bounded = solve_bounded_seed(*args, on_response=bounded_events.append)
    assert bounded["status"] == baseline["status"] == "margin_guarded_candidate"
    assert bounded["response_evaluations"] == baseline["response_evaluations"] <= 320
    assert bounded_events == baseline_events
    assert unbounded.choose_cached_seed is original_selector
    assert shared.isolated_margin_solver is original_isolator
    assert margin.select_evaluated_margin_candidate is original_ranking
    assert bounded["seed_preference"]["future_slack_increase"] <= MAX_FUTURE_SLACK_INCREASE + 1e-10
    selected = bounded["best_joint_safe"]
    audit_args = list(audit_args)
    audit_args[11] = targets
    checked = audit_two_step_witness(selected, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]


def test_bounded_adapter_retains_independent_audit_and_response_accounting():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    receipt, events = {}, []
    adapter = BoundedSeedPreferredAudit(receipt)
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)), on_response=events.append)
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["selection_scope"] == "margin_two_step"
    assert receipt["shared_shift_traces"][0]["audit_status"] == "passed"
    assert receipt["seed_preference_traces"][0]["extra_model_responses"] == 0
    assert events.count("attempted") == events.count("completed") == witness["response_evaluations"] <= 320
