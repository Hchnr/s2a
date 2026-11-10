"""A fallback may consume only a completed, audited first-pose plan."""

import numpy as np
import pytest

from s2a.research.joint_control.preview import causal_shift_fallback as fallback
from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped
from s2a.research.trajectory_search.adaptive_plan_history import AdaptivePlanHistory
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


def no_witness(count=5):
    return dict(
        status="two_step_no_witness",
        response_evaluations=count,
        initialization=dict(start_lp_attempts=0),
        search_phases=[dict(phase="two_point_feasibility", iterations=1)],
        first_stage_search_objective=dict(
            normalized_guard=1e-4, scale=1e4, acceptance_caps_changed=False, future_stage_changed=False
        ),
    )


def attach_completed(adapter, seed, previous):
    history = AdaptivePlanHistory()
    history.completed_periods = 1
    history.last_plan = np.vstack((previous, seed[:7]))
    history.stage_plan(seed)
    adapter.attach_history(history)
    return history


def test_completed_shift_can_pass_first_pose_with_bad_second_pose():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.1
    receipt = {}
    adapter = fallback.CausalShiftFallbackAudit(receipt, solver=lambda *_a, **_kw: no_witness())
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert witness["witness_found"] and witness["response_evaluations"] == 6
    assert witness["status"] == "shifted_first_pose_candidate"
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["first_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]
    assert checked["selection_scope"] == "shifted_plan_first_pose" and checked["hard_pose_prefix"] == 1
    assert np.linalg.norm(checked["tcp_errors"][1, :3]) > 0.01
    trace = receipt["causal_shift_fallback_traces"][0]
    assert trace["fallback_rollouts"] == 1 and trace["selected_scope"] == "first_only"
    assert trace["audit_status"] == "passed"
    with pytest.raises(ValueError, match="current selected plan"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))


def test_primary_two_point_candidate_has_priority_and_uses_both_pose_audit():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.015
    targets[-1, 0, 3] += 0.2
    receipt = {}
    adapter = fallback.CausalShiftFallbackAudit(receipt)
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert witness["status"] == "guarded_two_step_candidate"
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["prefix_pose_caps_satisfied"] and checked["hard_pose_prefix"] == 2
    assert receipt["causal_shift_fallback_traces"][0]["fallback_rollouts"] == 0


def test_failed_primary_audit_cannot_trigger_the_fallback():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.015
    adapter = fallback.CausalShiftFallbackAudit(
        {}, dual_auditor=lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("primary audit failure"))
    )
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    with pytest.raises(ValueError, match="primary audit failure"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert adapter.traces[-1]["audit_status"] == "rejected"
    assert adapter.traces[-1]["fallback_rollouts"] == 0


def test_uncompleted_or_mismatched_history_cannot_supply_a_fallback():
    args, _audit_args, _targets = solve_and_audit_inputs()
    adapter = fallback.CausalShiftFallbackAudit({}, solver=lambda *_a, **_kw: no_witness())
    history = AdaptivePlanHistory()
    history.stage_plan(args[2])
    adapter.attach_history(history)
    result = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert not result["witness_found"] and adapter.traces[-1]["fallback_status"] == "no_completed_plan"
    history.completed_periods = 1
    history.last_plan = np.vstack((np.ones(6) * 0.01, args[2][:7]))
    with pytest.raises(ValueError, match="last dispatched command"):
        adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))


def test_full_response_budget_does_not_launch_fallback(monkeypatch):
    args, _audit_args, _targets = solve_and_audit_inputs()
    adapter = fallback.CausalShiftFallbackAudit({}, solver=lambda *_a, **_kw: no_witness(320))
    attach_completed(adapter, args[2], args[3])
    monkeypatch.setattr(fallback, "rollout_joint_preview", lambda *_a, **_kw: pytest.fail("extra rollout"))
    result = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert not result["witness_found"] and result["response_evaluations"] == 320
    assert adapter.traces[-1]["fallback_status"] == "response_budget_exhausted"


def test_failed_fallback_audit_stops_the_transaction_before_dispatch(monkeypatch):
    adapter = fallback.CausalShiftFallbackAudit(
        {},
        solver=lambda *_a, **_kw: no_witness(),
        first_auditor=lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("injected audit failure")),
    )
    transaction, inner_history, receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    previous = arguments[8]
    seed = np.tile([0.01, 0, 0, 0.001, 0, 0], (8, 1))
    transaction.history.completed_periods = 1
    transaction.history.last_plan = np.vstack((previous, seed[:7]))
    adapter.attach_history(transaction.history)
    with pytest.raises(QualitySelectionStopped, match="injected audit failure"):
        transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert transaction.history.pending_plan is None and inner_history.snapshot()["completed_periods"] == 0
    assert receipt["quality_refiner"]["witness_audits_attempted"] == 1
    assert adapter.traces[-1]["audit_status"] == "rejected"


def test_checked_fallback_commits_only_after_its_completed_observation(monkeypatch):
    adapter = fallback.CausalShiftFallbackAudit({}, solver=lambda *_a, **_kw: no_witness())
    transaction, inner_history, receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    previous = arguments[8]
    seed = np.tile([0.01, 0, 0, 0.001, 0, 0], (8, 1))
    transaction.history.completed_periods = 1
    transaction.history.last_plan = np.vstack((previous, seed[:7]))
    adapter.attach_history(transaction.history)
    arguments[4][1, 0, 3] += 0.1
    result = transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert result["quality"]["verification"]["selection_scope"] == "shifted_plan_first_pose"
    assert receipt["quality_refiner"]["witness_audits_passed"] == 1
    assert inner_history.snapshot()["completed_periods"] == 0
    np.testing.assert_array_equal(transaction.history.last_plan[0], previous)
    transaction.commit(result["command"])
    inner_history.complete(10, np.zeros((25, 7)))
    transaction.observe(np.eye(4), np.zeros(6))
    assert inner_history.snapshot()["completed_periods"] == 1
    np.testing.assert_array_equal(transaction.history.last_plan, result["planned_commands"])


def test_fallback_rejects_a_first_pose_outside_original_threshold():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 0.03
    adapter = fallback.CausalShiftFallbackAudit({}, solver=lambda *_a, **_kw: no_witness())
    attach_completed(adapter, args[2], args[3])
    result = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert not result["witness_found"] and result["response_evaluations"] == 6
    assert adapter.traces[-1]["fallback_status"] == "model_safety_or_first_pose_rejected"
