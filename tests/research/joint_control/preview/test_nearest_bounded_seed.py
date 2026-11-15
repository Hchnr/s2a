"""Bounded cache selection must preserve the original independent gates."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.bounded_seed_preference import choose_bounded_seed
from s2a.research.joint_control.preview.nearest_bounded_seed import (
    NearestBoundedSeedAudit,
    choose_nearest_bounded_seed,
    solve_nearest_bounded_seed,
)
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import margin_guarded_fallback as shared
from s2a.research.trajectory_search import margin_two_step_search as margin
from tests.research.joint_control.preview.test_cached_seed_preference import candidate
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_shared_shift_fallback import attach_completed


def test_closest_intermediate_plan_respects_future_bound_without_interpolation():
    seed = np.full((8, 6), 0.1)
    default, intermediate, expensive = candidate(0.0, 0.1), candidate(0.06, 0.25), candidate(0.1, 0.5)
    candidates = [expensive, default, intermediate]
    old, _ = choose_bounded_seed(candidates, seed)
    chosen, trace = choose_nearest_bounded_seed(candidates, seed)
    assert old is default and chosen is intermediate
    assert trace["seed_blocked_by_future_slack"] is True and trace["changed_from_bounded"] is True
    assert trace["selected_changed"] is True and trace["selected_seed"] is False
    assert trace["bounded_candidates"] == 2 and trace["future_slack_increase"] == pytest.approx(0.15)
    assert trace["selected_seed_distance_squared"] < trace["bounded_seed_distance_squared"]
    assert trace["extra_model_responses"] == 0


def test_exact_seed_and_deterministic_ties_use_only_eligible_candidates():
    seed = np.full((8, 6), 0.1)
    default, exact = candidate(0.0, 0.1), candidate(0.1, 0.4)
    chosen, trace = choose_nearest_bounded_seed([default, exact], seed)
    assert chosen is exact and trace["selected_seed"] is True
    # The seed need not have been evaluated: compare cached candidates only.
    lower, upper = candidate(0.0, 0.0), candidate(0.2, 0.1)
    chosen, trace = choose_nearest_bounded_seed([upper, lower], seed)
    assert chosen is lower and trace["seed_in_eligible_cache"] is False
    with pytest.raises(ValueError, match="margins"):
        choose_nearest_bounded_seed([default, candidate(0.1, 0.2, position=0.0096)], seed)
    with pytest.raises(ValueError, match="nonempty"):
        choose_nearest_bounded_seed([], seed)


def test_nearest_solver_and_adapter_keep_search_and_independent_witness():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    baseline_events, candidate_events = [], []
    original_selector = margin.select_evaluated_margin_candidate
    original_isolator = shared.isolated_margin_solver
    baseline = shared.solve_margin_guarded_fallback(*args, on_response=baseline_events.append)
    result = solve_nearest_bounded_seed(*args, on_response=candidate_events.append)
    assert baseline["status"] == result["status"] == "margin_guarded_candidate"
    assert baseline_events == candidate_events
    assert baseline["response_evaluations"] == result["response_evaluations"] <= 320
    assert shared.isolated_margin_solver is original_isolator
    assert margin.select_evaluated_margin_candidate is original_selector
    trace = result["seed_preference"]
    assert trace["future_slack_increase"] <= 0.3 + 1e-10
    assert trace["selected_seed_distance_squared"] <= trace["bounded_seed_distance_squared"] + 1e-12
    direct_audit_args = list(audit_args)
    direct_audit_args[11] = targets
    verified = audit_two_step_witness(
        result["best_joint_safe"], *direct_audit_args, past_inner_error_rad=np.zeros((25, 7))
    )
    assert verified["passed"] and verified["prefix_pose_caps_satisfied"] and verified["braking_constraint_satisfied"]
    receipt, events = {}, []
    adapter = NearestBoundedSeedAudit(receipt)
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)), on_response=events.append)
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["selection_scope"] == "margin_two_step"
    assert receipt["seed_preference_traces"][0]["extra_model_responses"] == 0
    assert events.count("attempted") == events.count("completed") == witness["response_evaluations"] <= 320
