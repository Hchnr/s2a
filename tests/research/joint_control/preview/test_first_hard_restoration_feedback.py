"""The new restoration transaction records checked candidate evidence."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.first_hard_restoration_feedback import FirstHardRestorationSolverAudit
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


def test_audited_candidate_retains_first_and_future_quality_evidence():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.3
    receipt = {}
    adapter = FirstHardRestorationSolverAudit(receipt)
    witness = adapter.solve(*args)
    assert witness["witness_found"]
    verified = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert verified["passed"] and verified["first_pose_caps_satisfied"]
    trace = receipt["first_hard_restoration_traces"][0]
    assert trace["audit_status"] == "passed" and trace["candidate"]["first_pose_optimized_hard"]
    assert trace["candidate"]["future_quality_slack"] > 0
    assert trace["audit_detail"]["first_pose_caps_satisfied"] is True
    with pytest.raises(ValueError, match="current solver witness"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert trace["audit_status"] == "passed"


def test_failed_independent_audit_records_the_rejection_without_dispatch():
    args, audit_args, _targets = solve_and_audit_inputs()
    receipt = {}
    adapter = FirstHardRestorationSolverAudit(
        receipt, auditor=lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("injected audit failure"))
    )
    witness = adapter.solve(*args)
    assert witness["witness_found"]
    with pytest.raises(ValueError, match="injected audit failure"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    trace = receipt["first_hard_restoration_traces"][0]
    assert trace["audit_status"] == "rejected" and trace["audit_detail"]["message"] == "injected audit failure"
    assert trace["candidate"]["planned_commands"].shape == (8, 6)


def test_first_hard_integrates_with_the_existing_feedback_transaction(monkeypatch):
    trace_receipt = {}
    adapter = FirstHardRestorationSolverAudit(trace_receipt)
    transaction, history, receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    arguments[4][7, 0, 3] += 0.1
    result = transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert result["quality"]["verification"]["first_pose_caps_satisfied"]
    assert receipt["quality_refiner"]["witness_audits_passed"] == 1
    assert trace_receipt["first_hard_restoration_traces"][0]["audit_status"] == "passed"
    assert history.snapshot()["completed_periods"] == 0
