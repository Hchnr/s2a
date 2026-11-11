"""A feedback command requires an independent margin and safety audit."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.margin_two_step_feedback import MarginTwoStepFeedbackSolverAudit
from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


def test_margin_feedback_candidate_passes_independent_two_pose_and_safety_audit():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.2
    receipt = {}
    adapter = MarginTwoStepFeedbackSolverAudit(receipt)
    witness = adapter.solve(*args)
    assert witness["witness_found"]
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]
    assert checked["model_margin_satisfied"] is True
    trace = receipt["margin_two_step_traces"][0]
    assert trace["model_position_margin_m"] == 0.0005 and trace["model_rotation_margin_rad"] == 0.003
    assert trace["audit_status"] == "passed" and trace["response_evaluations"] <= 320


def test_independent_margin_mismatch_rejects_candidate_before_dispatch():
    args, audit_args, _targets = solve_and_audit_inputs()

    def mismatched_auditor(*a, **kw):
        checked = audit_two_step_witness(*a, **kw)
        checked["tcp_errors"][1, 0] = 0.0097
        return checked

    receipt = {}
    adapter = MarginTwoStepFeedbackSolverAudit(receipt, auditor=mismatched_auditor)
    witness = adapter.solve(*args)
    with pytest.raises(ValueError, match="Independent two-point reconstruction"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert receipt["margin_two_step_traces"][0]["audit_status"] == "rejected"
    assert receipt["two_step_feedback_traces"][0]["audit_status"] == "rejected"


def test_margin_feedback_stops_before_dispatch_on_unreachable_reference(monkeypatch):
    receipt = {}
    adapter = MarginTwoStepFeedbackSolverAudit(receipt)
    transaction, history, accounting, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    arguments[4][1, 0, 3] += 1.0
    with pytest.raises(QualitySelectionStopped, match="two_step_margin_no_witness"):
        transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert transaction.history.pending_plan is None and history.snapshot()["completed_periods"] == 0
    assert accounting["quality_refiner"]["witness_audits_attempted"] == 0
    assert receipt["margin_two_step_traces"][0]["audit_status"] == "not_attempted"


def test_missing_margin_metadata_is_rejected_before_dispatch():
    args, _audit_args, _targets = solve_and_audit_inputs()
    adapter = MarginTwoStepFeedbackSolverAudit({}, solver=lambda *_a, **_kw: {"margin_search": {}})
    with pytest.raises(ValueError, match="Exact model margins"):
        adapter.solve(*args)
