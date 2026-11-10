"""Joint-safe restoration is not an eight-point pose-quality witness."""

import copy

import numpy as np
import pytest

from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped
from s2a.research.joint_control.preview.restoration_horizon_feedback import RestorationHorizonSolverAudit
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


def test_restoration_dispatch_candidate_reports_future_quality_gap():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.1
    adapter = RestorationHorizonSolverAudit()
    witness = adapter.solve(*args)
    assert witness["witness_found"] and witness["quality_slack"] > 0
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["first_pose_caps_enforced"] and checked["first_pose_caps_satisfied"]
    assert checked["braking_constraint_enforced"] and checked["braking_constraint_satisfied"]
    assert not checked["all_pose_caps_enforced"] and not checked["all_pose_caps_satisfied"]
    assert checked["quality_slack"] > 0 and checked["selection_scope"] == "joint_and_first_pose"


def test_restoration_rejects_first_pose_failure_and_consumes_transaction():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.3
    adapter = RestorationHorizonSolverAudit()
    witness = adapter.solve(*args)
    assert witness["witness_found"]
    with pytest.raises(ValueError, match="first pose"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    with pytest.raises(ValueError, match="current solver witness"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))


def test_restoration_rejects_changed_candidate_and_late_target():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[-1, 0, 3] += 0.1
    adapter = RestorationHorizonSolverAudit()
    witness = adapter.solve(*args)
    with pytest.raises(ValueError, match="current solver witness"):
        adapter.audit(copy.deepcopy(witness), *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    witness["planned_commands"][0, 0] += 2
    with pytest.raises(ValueError):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))

    adapter = RestorationHorizonSolverAudit()
    witness = adapter.solve(*args)
    witness["tcp_errors"][-1, 0] += 0.1
    with pytest.raises(AssertionError):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))


def test_restoration_cannot_disable_braking():
    args, _audit_args, _targets = solve_and_audit_inputs()
    with pytest.raises(ValueError, match="braking"):
        RestorationHorizonSolverAudit().solve(*args, enforce_braking=False)


def test_restoration_integrates_with_feedback_without_promoting_future_quality(monkeypatch):
    adapter = RestorationHorizonSolverAudit()
    transaction, history, receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    arguments[4][7, 0, 3] += 0.1
    result = transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    audit = result["quality"]["verification"]
    assert audit["first_pose_caps_satisfied"] and not audit["all_pose_caps_satisfied"]
    assert audit["quality_slack"] > 0 and not audit["all_pose_caps_enforced"]
    assert receipt["quality_refiner"]["witness_audits_passed"] == 1
    assert history.snapshot()["completed_periods"] == 0


def test_restoration_first_pose_failure_does_not_stage_a_plan(monkeypatch):
    adapter = RestorationHorizonSolverAudit()
    transaction, history, _receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    arguments[4][7, 0, 3] += 0.3
    with pytest.raises(QualitySelectionStopped, match="first pose"):
        transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert transaction.history.pending_plan is None and history.snapshot()["completed_periods"] == 0
