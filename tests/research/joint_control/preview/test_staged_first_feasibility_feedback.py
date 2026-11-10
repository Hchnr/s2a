"""Staged candidates retain phase costs and require the original audit."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.staged_first_feasibility_feedback import StagedFirstFeasibilitySolverAudit
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


def test_staged_candidate_records_both_stages_and_original_audit():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 0.02
    targets[-1, 0, 3] += 0.1
    receipt = {}
    adapter = StagedFirstFeasibilitySolverAudit(receipt)
    witness = adapter.solve(*args)
    assert witness["witness_found"]
    check = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert check["passed"] and check["first_pose_caps_satisfied"] and check["braking_constraint_satisfied"]
    trace = receipt["staged_first_feasibility_traces"][0]
    assert trace["phases"][0]["phase"] == "first_feasibility" and trace["response_evaluations"] <= 320
    assert receipt["first_hard_restoration_traces"][0]["audit_status"] == "passed"
    assert receipt["feasible_start_traces"][0]["actual_model_rollouts"] == trace["response_evaluations"]


def test_staged_feedback_does_not_advance_history_before_dispatch(monkeypatch):
    trace_receipt = {}
    adapter = StagedFirstFeasibilitySolverAudit(trace_receipt)
    transaction, history, receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    result = transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert result["quality"]["verification"]["first_pose_caps_satisfied"]
    assert receipt["quality_refiner"]["witness_audits_passed"] == 1
    assert history.snapshot()["completed_periods"] == 0


def test_unbounded_or_misordered_stage_receipt_is_rejected():
    args, _audit_args, _targets = solve_and_audit_inputs()
    receipt = {}

    def fake(*_args, **_kwargs):
        return {
            "status": "fake",
            "response_evaluations": 1,
            "search_phases": [{"phase": "first_feasibility", "iterations": 41}],
        }

    adapter = StagedFirstFeasibilitySolverAudit(receipt, solver=fake)
    with pytest.raises(ValueError, match="iteration budget"):
        adapter.solve(*args)
