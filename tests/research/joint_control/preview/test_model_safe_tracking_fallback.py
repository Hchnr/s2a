"""A degraded fallback must never masquerade as an original pose witness."""

import copy

import numpy as np
import pytest

from s2a.research.joint_control.preview import model_safe_tracking_fallback as fallback
from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped
from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture
from tests.research.joint_control.preview.test_shared_shift_fallback import (
    attach_completed,
    fake_solver,
    no_shared_candidate,
)


def adapter_with_no_strict_candidate(count=5, **options):
    receipt = {}
    primary = SharedShiftFallbackAudit(receipt, solver=fake_solver(no_shared_candidate(count)))
    return fallback.ModelSafeTrackingFallback(receipt, primary=primary, **options), receipt


def test_strict_candidate_has_priority_and_keeps_original_audit():
    args, audit_args, _targets = solve_and_audit_inputs()
    adapter = fallback.ModelSafeTrackingFallback({})
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert witness["witness_found"] and witness["selection_scope"] == "margin_two_step"
    verified = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert verified["passed"] and verified["prefix_pose_caps_satisfied"]
    assert adapter.traces[-1]["selected_scope"] == "strict" and adapter.traces[-1]["extra_rollouts"] == 0


def test_bounded_fallback_rechecks_and_audits_safety_without_claiming_pose_quality():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 0.015
    adapter, _receipt = adapter_with_no_strict_candidate()
    attach_completed(adapter, args[2], args[3])
    events = []
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)), on_response=events.append)
    assert witness["status"] == witness["selection_scope"] == fallback.STATUS
    assert witness["hard_pose_prefix"] == 0 and witness["response_evaluations"] == 7
    assert events.count("attempted") == events.count("completed") == 7
    verified = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert verified["passed"] and verified["braking_constraint_enforced"]
    assert verified["first_pose_caps_enforced"] is verified["first_pose_caps_satisfied"] is False
    assert verified["original_pose_caps_enforced"] is False and verified["selection_scope"] == fallback.STATUS
    assert adapter.traces[-1]["audit_status"] == "passed"


def test_out_of_bound_and_exhausted_budget_do_not_dispatch():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 0.021
    adapter, _ = adapter_with_no_strict_candidate()
    attach_completed(adapter, args[2], args[3])
    result = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert not result["witness_found"] and adapter.traces[-1]["fallback_status"] == "first_pose_outside_fallback_bound"
    assert adapter.traces[-1]["extra_rollouts"] == 0

    args, _audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 0.015
    adapter, _ = adapter_with_no_strict_candidate(319)
    attach_completed(adapter, args[2], args[3])
    result = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert not result["witness_found"] and result["response_evaluations"] == 320
    assert adapter.traces[-1]["fallback_status"] == "response_budget_exhausted"


def test_changed_or_independently_rejected_fallback_stops_before_dispatch():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 0.015
    adapter, _ = adapter_with_no_strict_candidate()
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    with pytest.raises(ValueError, match="currently selected"):
        adapter.audit(copy.deepcopy(witness), *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    witness["planned_commands"][0, 0] += 0.2
    with pytest.raises((ValueError, AssertionError)):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert adapter.traces[-1]["audit_status"] == "rejected"


def test_failed_strict_audit_never_enters_degraded_fallback():
    args, audit_args, _targets = solve_and_audit_inputs()
    receipt = {}
    primary = SharedShiftFallbackAudit(
        receipt, dual_auditor=lambda *_a, **_k: (_ for _ in ()).throw(ValueError("strict audit"))
    )
    adapter = fallback.ModelSafeTrackingFallback(receipt, primary=primary)
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    with pytest.raises(ValueError, match="strict audit"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert adapter.traces[-1]["audit_status"] == "rejected" and adapter.traces[-1]["extra_rollouts"] == 0


def test_transaction_marks_degraded_scope_and_only_commits_completed_history(monkeypatch):
    adapter, receipt = adapter_with_no_strict_candidate()
    transaction, inner, _unused, args = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    original = transaction._refiner
    transaction._refiner = fallback.ScopedQualityFeedbackRefiner(
        inner,
        receipt,
        start_step=10,
        start_cursor=0,
        baseline=original.baseline,
        solver=adapter.solve,
        auditor=adapter.audit,
        max_quality_response_evaluations=320,
    )
    previous = args[8]
    seed = np.tile([0.01, 0, 0, 0.001, 0, 0], (8, 1))
    transaction.history.completed_periods = 1
    transaction.history.last_plan = np.vstack((previous, seed[:7]))
    adapter.attach_history(transaction.history)
    args[4][0, 0, 3] += 0.015
    result = transaction.prepare(*args, site_from_tcp=np.eye(4))
    assert result["route"] == fallback.STATUS and result["quality"]["original_pose_caps_enforced"] is False
    assert result["quality"]["verification"]["first_pose_caps_satisfied"] is False
    assert inner.snapshot()["completed_periods"] == 0
    np.testing.assert_array_equal(transaction.history.last_plan[0], previous)
    transaction.commit(result["command"])
    inner.complete(10, np.zeros((25, 7)))
    transaction.observe(np.eye(4), np.zeros(6))
    np.testing.assert_array_equal(transaction.history.last_plan, result["planned_commands"])


def test_degraded_audit_failure_is_a_stop_not_a_quality_witness(monkeypatch):
    adapter, receipt = adapter_with_no_strict_candidate(
        auditor=lambda *_a, **_k: (_ for _ in ()).throw(ValueError("safety audit"))
    )
    transaction, inner, _unused, args = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    seed = np.tile([0.01, 0, 0, 0.001, 0, 0], (8, 1))
    transaction.history.completed_periods = 1
    transaction.history.last_plan = np.vstack((args[8], seed[:7]))
    adapter.attach_history(transaction.history)
    args[4][0, 0, 3] += 0.015
    with pytest.raises(QualitySelectionStopped, match="safety audit"):
        transaction.prepare(*args, site_from_tcp=np.eye(4))
    assert transaction.history.pending_plan is None and inner.snapshot()["completed_periods"] == 0
    assert receipt["model_safe_tracking_fallback_traces"][-1]["audit_status"] == "rejected"
