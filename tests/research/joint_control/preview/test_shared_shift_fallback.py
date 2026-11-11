"""Three-tier feedback must retain causal history, scope and common budgets."""

import copy

import numpy as np
import pytest

from s2a.research.joint_control.preview import shared_shift_fallback as combined
from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped
from s2a.research.trajectory_search.adaptive_plan_history import AdaptivePlanHistory
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs
from tests.research.joint_control.preview.test_quality_feedback_refiner import fixture as feedback_fixture


def no_shared_candidate(count=5, fallback_status="two_step_no_witness"):
    primary = max(count - 1, 0)
    return dict(
        status="margin_guarded_no_witness",
        response_evaluations=count,
        initialization=dict(start_lp_attempts=0, actual_model_rollouts=count),
        search_phases=[dict(phase="two_point_feasibility", iterations=1, search="margin")],
        shared_search=dict(
            primary_status="two_step_margin_no_witness",
            primary_responses=primary,
            primary_phases=[dict(phase="two_point_feasibility", iterations=1)],
            primary_feasibility_iterations=1,
            primary_total_iterations=1,
            fallback_status=fallback_status,
            fallback_responses=count - primary,
            fallback_phases=[dict(phase="two_point_feasibility", iterations=1)],
            fallback_initializer_reused=True,
            fallback_lp_attempts=0,
            selected_scope=None,
            response_budget=320,
            feasibility_iteration_budget=60,
            total_iteration_budget=120,
            start_lp_budget=2,
            optimizer_calls=[dict(phase="two_point_feasibility", limit=59, iterations=1)],
        ),
    )


def fake_solver(result):
    def solve(*_args, on_response=None, **_kwargs):
        if on_response is not None:
            for _ in range(result["response_evaluations"]):
                on_response("attempted")
                on_response("completed")
        return copy.deepcopy(result)

    return solve


def attach_completed(adapter, seed, previous):
    history = AdaptivePlanHistory()
    history.completed_periods = 1
    history.last_plan = np.vstack((previous, seed[:7]))
    history.stage_plan(seed)
    adapter.attach_history(history)
    return history


@pytest.mark.parametrize("offset,scope", [(0.0, "margin_two_step"), (0.015, "guarded_two_step")])
def test_dual_candidates_have_priority_and_scope_specific_audit(offset, scope):
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += offset
    targets[-1, 0, 3] += 0.2
    receipt = {}
    adapter = combined.SharedShiftFallbackAudit(receipt)
    attach_completed(adapter, args[2], args[3])
    events = []
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)), on_response=events.append)
    assert witness["witness_found"] and witness["selection_scope"] == scope
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["prefix_pose_caps_satisfied"] and checked["selection_scope"] == scope
    assert checked["model_margin_enforced"] is (scope == "margin_two_step")
    trace = receipt["shared_shift_traces"][0]
    assert trace["selected_scope"] == scope and trace["shifted_rollouts"] == 0 and trace["audit_status"] == "passed"
    assert events.count("attempted") == events.count("completed") == trace["response_evaluations"] <= 320


def test_completed_shift_first_pose_has_separate_audit_and_shared_response_charge():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.1
    receipt, events = {}, []
    adapter = combined.SharedShiftFallbackAudit(receipt, solver=fake_solver(no_shared_candidate()))
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)), on_response=events.append)
    assert witness["selection_scope"] == "shifted_plan_first_pose" and witness["response_evaluations"] == 6
    verified = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert verified["passed"] and verified["hard_pose_prefix"] == 1 and verified["braking_constraint_satisfied"]
    assert np.linalg.norm(verified["tcp_errors"][1, :3]) > 0.01
    trace = receipt["shared_shift_traces"][0]
    assert trace["shifted_rollouts"] == 1 and trace["audit_status"] == "passed"
    assert events.count("attempted") == events.count("completed") == 6


def test_failed_dual_audit_never_tries_the_shifted_plan():
    args, audit_args, _targets = solve_and_audit_inputs()
    receipt = {}
    adapter = combined.SharedShiftFallbackAudit(
        receipt, dual_auditor=lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("dual audit failure"))
    )
    attach_completed(adapter, args[2], args[3])
    witness = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    with pytest.raises(ValueError, match="dual audit failure"):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert receipt["shared_shift_traces"][0]["audit_status"] == "rejected"
    assert receipt["shared_shift_traces"][0]["shifted_rollouts"] == 0


def test_missing_history_or_exhausted_response_budget_forbids_shift():
    args, _audit_args, _targets = solve_and_audit_inputs()
    adapter = combined.SharedShiftFallbackAudit({}, solver=fake_solver(no_shared_candidate()))
    history = AdaptivePlanHistory()
    history.stage_plan(args[2])
    adapter.attach_history(history)
    result = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert not result["witness_found"] and adapter.shared_traces[-1]["shifted_status"] == "no_completed_plan"
    full = combined.SharedShiftFallbackAudit({}, solver=fake_solver(no_shared_candidate(320)))
    attach_completed(full, args[2], args[3])
    result = full.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert not result["witness_found"] and result["response_evaluations"] == 320
    assert full.shared_traces[-1]["shifted_status"] == "response_budget_exhausted"


def test_interrupted_shared_search_cannot_enter_first_only_fallback():
    args, _audit_args, _targets = solve_and_audit_inputs()
    interrupted = no_shared_candidate(fallback_status="SharedIterationBudgetExhausted")
    adapter = combined.SharedShiftFallbackAudit({}, solver=fake_solver(interrupted))
    attach_completed(adapter, args[2], args[3])
    result = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert not result["witness_found"] and result["status"] == "shared_search_ineligible"
    assert adapter.shared_traces[-1]["shifted_rollouts"] == 0


def test_unfinished_guarded_phase_cannot_be_treated_as_a_normal_no_witness():
    args, _audit_args, _targets = solve_and_audit_inputs()
    interrupted = no_shared_candidate()
    interrupted["shared_search"]["optimizer_calls"][0]["iterations"] = None
    interrupted["shared_search"]["fallback_phases"][0]["iterations"] = None
    adapter = combined.SharedShiftFallbackAudit({}, solver=fake_solver(interrupted))
    attach_completed(adapter, args[2], args[3])
    result = adapter.solve(*args, past_inner_error_rad=np.zeros((25, 7)))
    assert not result["witness_found"] and result["status"] == "shared_search_ineligible"
    assert adapter.shared_traces[-1]["shifted_rollouts"] == 0


def test_tampered_shared_iteration_budget_is_rejected():
    result = no_shared_candidate()
    result["shared_search"]["optimizer_calls"][0]["limit"] = 60
    with pytest.raises(ValueError, match="iteration budget"):
        combined.validate_shared_result(result)


def test_failed_shift_audit_stops_transaction_without_committing_history(monkeypatch):
    adapter = combined.SharedShiftFallbackAudit(
        {},
        solver=fake_solver(no_shared_candidate()),
        first_auditor=lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("injected first audit failure")),
    )
    transaction, inner_history, receipt, arguments = feedback_fixture(
        monkeypatch, solver=adapter.solve, auditor=adapter.audit, max_quality_response_evaluations=320
    )
    previous = arguments[8]
    seed = np.tile([0.01, 0, 0, 0.001, 0, 0], (8, 1))
    transaction.history.completed_periods = 1
    transaction.history.last_plan = np.vstack((previous, seed[:7]))
    adapter.attach_history(transaction.history)
    with pytest.raises(QualitySelectionStopped, match="injected first audit failure"):
        transaction.prepare(*arguments, site_from_tcp=np.eye(4))
    assert transaction.history.pending_plan is None and inner_history.snapshot()["completed_periods"] == 0
    assert receipt["quality_refiner"]["witness_audits_attempted"] == 1
    assert adapter.shared_traces[-1]["audit_status"] == "rejected"


def test_audited_shift_enters_history_only_after_completed_observation(monkeypatch):
    adapter = combined.SharedShiftFallbackAudit({}, solver=fake_solver(no_shared_candidate()))
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
    np.testing.assert_array_equal(transaction.history.last_plan[0], previous)
    transaction.commit(result["command"])
    inner_history.complete(10, np.zeros((25, 7)))
    transaction.observe(np.eye(4), np.zeros(6))
    np.testing.assert_array_equal(transaction.history.last_plan, result["planned_commands"])
