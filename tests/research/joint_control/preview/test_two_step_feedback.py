"""Only independently audited two-point plans may advance feedback."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped
from s2a.research.joint_control.preview.two_step_feedback import TwoStepFeedbackSolverAudit
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


def test_two_step_feedback_retains_phase_budget_and_audits_both_points():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.015
    targets[-1, 0, 3] += 0.2
    receipt = {}
    adapter = TwoStepFeedbackSolverAudit(receipt)
    witness = adapter.solve(*args)
    assert witness["witness_found"]
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]
    trace = receipt["two_step_feedback_traces"][0]
    assert trace["phases"][0]["phase"] == "two_point_feasibility" and trace["response_evaluations"] <= 320
    assert trace["audit_status"] == receipt["first_hard_restoration_traces"][0]["audit_status"] == "passed"
    assert receipt["feasible_start_traces"][0]["actual_model_rollouts"] == trace["response_evaluations"]


def test_feedback_does_not_dispatch_when_second_point_is_unreachable(monkeypatch):
    trace_receipt = {}
    adapter = TwoStepFeedbackSolverAudit(trace_receipt)
    transaction, history, receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    arguments[4][1, 0, 3] += 1.0
    with pytest.raises(QualitySelectionStopped, match="two_step_no_witness"):
        transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert transaction.history.pending_plan is None and history.snapshot()["completed_periods"] == 0
    assert receipt["quality_refiner"]["witness_audits_attempted"] == 0
    assert trace_receipt["two_step_feedback_traces"][0]["audit_status"] == "not_attempted"


def test_invalid_two_step_iteration_receipt_is_rejected_before_dispatch():
    args, _audit_args, _targets = solve_and_audit_inputs()

    def fake(*_args, **_kwargs):
        return {
            "status": "fake",
            "response_evaluations": 1,
            "initialization": {"start_lp_attempts": 0},
            "search_phases": [{"phase": "two_point_feasibility", "iterations": 61}],
        }

    adapter = TwoStepFeedbackSolverAudit({}, solver=fake)
    with pytest.raises(ValueError, match="iteration budget"):
        adapter.solve(*args)
