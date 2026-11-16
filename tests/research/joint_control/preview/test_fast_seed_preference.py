"""Early seed selection cannot bypass constraints, budget or witness audit."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.cached_seed_preference import solve_prefer_cached_seed
from s2a.research.joint_control.preview.fast_seed_preference import FastSeedPreferredAudit, solve_fast_preferred_seed
from s2a.research.joint_control.preview.shared_shift_fallback import validate_shared_result
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import margin_guarded_fallback as shared
from s2a.research.trajectory_search import margin_two_step_search as margin
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_shared_shift_fallback import attach_completed


def test_qualified_seed_is_the_same_action_and_keeps_independent_witness():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    baseline_events, early_events = [], []
    old_isolator, old_selector = shared.isolated_margin_solver, margin.select_evaluated_margin_candidate
    baseline = solve_prefer_cached_seed(*args, on_response=baseline_events.append)
    result = solve_fast_preferred_seed(*args, on_response=early_events.append)
    assert baseline["seed_preference"]["seed_in_eligible_cache"] is True
    assert result["fast_seed"]["early_selected"] is True
    assert result["fast_seed"]["baseline_future_slack"] is None
    assert result["fast_seed"]["future_slack_increase"] is None
    assert result["response_evaluations"] == validate_shared_result(result) == 1
    assert early_events == ["attempted", "completed"]
    assert result["search_phases"] == []
    assert shared.isolated_margin_solver is old_isolator and margin.select_evaluated_margin_candidate is old_selector
    witness = result["best_joint_safe"]
    np.testing.assert_array_equal(witness["planned_commands"], baseline["best_joint_safe"]["planned_commands"])
    np.testing.assert_array_equal(witness["planned_commands"], args[2])
    audit_args = list(audit_args)
    audit_args[11] = targets
    checked = audit_two_step_witness(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]


def test_rejected_seed_falls_back_without_an_extra_model_response():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[:2, 0, 3] += 0.015
    baseline_events, new_events = [], []
    baseline = solve_prefer_cached_seed(*args, on_response=baseline_events.append)
    result = solve_fast_preferred_seed(*args, on_response=new_events.append)
    assert result["fast_seed"]["early_selected"] is False
    assert result["fast_seed"]["original_solver_continued"] is True
    assert result["fast_seed"]["additional_model_responses"] == 0
    assert result["status"] == baseline["status"]
    assert result["response_evaluations"] == baseline["response_evaluations"]
    assert new_events == baseline_events
    assert result["seed_preference"] == baseline["seed_preference"]
    assert validate_shared_result(result) == result["response_evaluations"] <= 320
    if "best_joint_safe" in baseline:
        np.testing.assert_array_equal(
            result["best_joint_safe"]["planned_commands"], baseline["best_joint_safe"]["planned_commands"]
        )


def test_fast_adapter_audits_before_a_selected_action_can_be_reused():
    args, audit_args, _targets = solve_and_audit_inputs()
    receipt, events = {}, []
    adapter = FastSeedPreferredAudit(receipt)
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)), on_response=events.append)
    assert receipt["fast_seed_traces"][0]["early_selected"] is True
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["selection_scope"] == "margin_two_step"
    assert receipt["shared_shift_traces"][0]["audit_status"] == "passed"
    assert events == ["attempted", "completed"]
    with pytest.raises(ValueError, match="current selected plan"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))


def test_fast_path_keeps_original_command_and_refresh_guards():
    args, _audit_args, _targets = solve_and_audit_inputs()
    with pytest.raises(ValueError, match="cannot be combined"):
        solve_fast_preferred_seed(*args, prefer_refresh=True)
    changed = list(args)
    changed[2] = changed[2].copy()
    changed[2][0, 0] = 1.5
    with pytest.raises(ValueError, match="original command box and rate"):
        solve_fast_preferred_seed(*changed)


def test_initial_joint_buffer_cannot_be_bypassed_by_a_good_seed():
    args, _audit_args, _targets = solve_and_audit_inputs()
    changed = list(args)
    changed[9] = changed[9].copy()
    changed[9][0] = changed[0]["q"][0] - 0.04
    baseline_events, new_events = [], []
    baseline = solve_prefer_cached_seed(*changed, on_response=baseline_events.append)
    result = solve_fast_preferred_seed(*changed, on_response=new_events.append)
    assert result["fast_seed"]["early_selected"] is False
    assert result["fast_seed"]["seed_probe_reached"] is False
    assert result["status"] == baseline["status"] == "margin_guarded_no_witness"
    assert result["shared_search"]["primary_status"] == "initial_outside_requested_interior"
    assert new_events == baseline_events == ["attempted", "completed"]


def test_invalid_completed_inner_error_is_not_hidden_by_early_selection():
    args, _audit_args, _targets = solve_and_audit_inputs()
    history = np.zeros((25, 7))
    history[0, 0] = 0.1
    events = []
    with pytest.raises(ValueError, match="nonnegative completed inner error"):
        solve_fast_preferred_seed(*args, past_inner_error_rad=history, on_response=events.append)
    assert events == ["attempted", "completed"]
