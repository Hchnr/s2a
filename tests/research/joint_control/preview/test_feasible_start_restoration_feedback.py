"""A feasible initializer must preserve the original audited transaction."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.feasible_start_restoration_feedback import FeasibleStartRestorationSolverAudit
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


def test_safe_seed_records_single_budget_and_passes_existing_audit():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.3
    receipt = {}
    adapter = FeasibleStartRestorationSolverAudit(receipt)
    witness = adapter.solve(*args)
    assert witness["witness_found"]
    result = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert result["passed"] and result["first_pose_caps_satisfied"] and result["braking_constraint_satisfied"]
    start = receipt["feasible_start_traces"][0]
    assert start["seed_linear_safe"] and start["start_lp_attempts"] == 0 and start["seed_rollout_reused"]
    assert start["actual_model_rollouts"] == witness["response_evaluations"]
    assert receipt["first_hard_restoration_traces"][0]["audit_status"] == "passed"


def test_initializer_accounting_failure_is_rejected_before_dispatch():
    args, _audit_args, _targets = solve_and_audit_inputs()
    receipt = {}
    adapter = FeasibleStartRestorationSolverAudit(
        receipt,
        solver=lambda *_a, **_kw: {
            "status": "fake",
            "response_evaluations": 1,
            "initialization": {"start_lp_attempts": 3, "actual_model_rollouts": 1},
        },
    )
    with pytest.raises(ValueError, match="at most two LPs"):
        adapter.solve(*args)
    assert receipt["feasible_start_traces"] == []


def test_candidate_uses_quality_transaction_and_past_only_history(monkeypatch):
    trace_receipt = {}
    adapter = FeasibleStartRestorationSolverAudit(trace_receipt)
    transaction, history, receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    arguments[4][7, 0, 3] += 0.1
    result = transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert result["quality"]["verification"]["first_pose_caps_satisfied"]
    assert receipt["quality_refiner"]["witness_audits_passed"] == 1
    assert trace_receipt["feasible_start_traces"][0]["actual_model_rollouts"] <= 320
    assert history.snapshot()["completed_periods"] == 0
