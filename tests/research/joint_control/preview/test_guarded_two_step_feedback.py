"""A scaled search must retain the original audited feedback transaction."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.guarded_two_step_feedback import GuardedTwoStepFeedbackSolverAudit
from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


def test_guarded_candidate_preserves_two_point_and_safety_audit():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.015
    targets[-1, 0, 3] += 0.2
    receipt = {}
    adapter = GuardedTwoStepFeedbackSolverAudit(receipt)
    witness = adapter.solve(*args)
    assert witness["witness_found"]
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]
    trace = receipt["guarded_two_step_traces"][0]
    assert trace["normalized_guard"] == 1e-4 and trace["scale"] == 1e4
    assert trace["acceptance_caps_changed"] is False and trace["response_evaluations"] <= 320
    assert receipt["two_step_feedback_traces"][0]["audit_status"] == "passed"


def test_guarded_feedback_never_dispatches_unreachable_second_pose(monkeypatch):
    trace = {}
    adapter = GuardedTwoStepFeedbackSolverAudit(trace)
    transaction, history, receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    arguments[4][1, 0, 3] += 1.0
    with pytest.raises(QualitySelectionStopped, match="two_step_no_witness"):
        transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert transaction.history.pending_plan is None and history.snapshot()["completed_periods"] == 0
    assert receipt["quality_refiner"]["witness_audits_attempted"] == 0


def test_guard_metadata_mismatch_is_rejected_before_dispatch():
    args, _audit_args, _targets = solve_and_audit_inputs()
    adapter = GuardedTwoStepFeedbackSolverAudit({}, solver=lambda *_a, **_k: {"first_stage_search_objective": {}})
    with pytest.raises(ValueError, match="unchanged acceptance"):
        adapter.solve(*args)
