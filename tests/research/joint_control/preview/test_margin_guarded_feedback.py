"""The selected scope must pass independent audit before feedback dispatch."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.margin_guarded_feedback import MarginGuardedFeedbackAudit
from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


@pytest.mark.parametrize("offset,scope", [(0.0, "margin_two_step"), (0.015, "guarded_two_step")])
def test_selected_scope_passes_its_independent_two_point_audit(offset, scope):
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += offset
    targets[-1, 0, 3] += 0.2
    receipt = {}
    adapter = MarginGuardedFeedbackAudit(receipt)
    witness = adapter.solve(*args)
    assert witness["witness_found"] and witness["selection_scope"] == scope
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]
    assert checked["selection_scope"] == scope
    assert checked["model_margin_enforced"] is (scope == "margin_two_step")
    trace = receipt["margin_guarded_traces"][0]
    assert trace["audit_status"] == "passed" and trace["selected_scope"] == scope
    assert trace["primary_responses"] + trace["fallback_responses"] == trace["response_evaluations"] <= 320


def test_failed_margin_audit_does_not_try_a_guarded_fallback():
    args, audit_args, _targets = solve_and_audit_inputs()

    def mismatched_auditor(*a, **kw):
        checked = audit_two_step_witness(*a, **kw)
        checked["tcp_errors"][1, 0] = 0.0097
        return checked

    receipt = {}
    adapter = MarginGuardedFeedbackAudit(receipt, auditor=mismatched_auditor)
    witness = adapter.solve(*args)
    assert witness["selection_scope"] == "margin_two_step"
    with pytest.raises(ValueError, match="Independent reconstruction"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    trace = receipt["margin_guarded_traces"][0]
    assert trace["audit_status"] == "rejected" and trace["fallback_status"] == "not_attempted"
    assert trace["fallback_responses"] == 0


def test_unreachable_reference_stops_before_dispatch(monkeypatch):
    receipt = {}
    adapter = MarginGuardedFeedbackAudit(receipt)
    transaction, history, accounting, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    arguments[4][1, 0, 3] += 1.0
    with pytest.raises(QualitySelectionStopped, match="margin_guarded_no_witness"):
        transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert transaction.history.pending_plan is None and history.snapshot()["completed_periods"] == 0
    assert accounting["quality_refiner"]["witness_audits_attempted"] == 0


def test_missing_budget_metadata_is_rejected_before_dispatch():
    args, _audit_args, _targets = solve_and_audit_inputs()
    adapter = MarginGuardedFeedbackAudit({}, solver=lambda *_a, **_kw: {"shared_search": {}})
    with pytest.raises(ValueError, match="Shared response"):
        adapter.solve(*args)
