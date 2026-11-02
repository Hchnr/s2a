"""Hand-certified cap ledgers, deliberate corruption and independent arithmetic."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import s2a.research.trajectory_search.cap_distance_objective as production
from s2a.research.trajectory_search.objective_search_driver import search
from scripts.tracking.trajectory_search.cap_distance_audit_policy import cap_loss, projection_model
from scripts.tracking.trajectory_search.cap_distance_fresh_audit import verify_cap_fresh_decisions
from scripts.tracking.trajectory_search.cap_distance_search_audit import audit_cap_search
from tests.research.trajectory_search.test_state_vector_dual_recovery_audit import annotate
from tests.research.trajectory_search.test_state_vector_gn_search_audit import (
    hand_certificate,
    restoration_inputs,
    vector_inputs,
)


def archive(evaluate, arguments):
    Js, Jq, Je, initial, previous, lo, hi, cfg = arguments

    def solve(B, g, A, b, initial_t, **kwargs):
        certificate = hand_certificate(B, g, A, b, initial_t, **kwargs)
        return annotate(certificate, A, b, kwargs["dual_recovery_mode"], kwargs["objective_divisor"])

    return search(
        evaluate,
        initial,
        Js,
        Jq,
        Je,
        previous,
        lo,
        hi,
        cfg,
        objective=production.CapDistanceObjective(),
        qp_solver=solve,
    )


@pytest.fixture
def saved():
    evaluate, arguments, _ = vector_inputs(curvature=20.0)
    answer = archive(evaluate, arguments)
    return answer, arguments


def fresh(answer):
    records = answer["evaluated_candidates"]
    return [r["rollout"] for r in records], [
        dict(
            joint_feasible=r["audit"]["feasible"],
            safety_slack=r["safety_slack"].copy(),
            normalized_squared_errors=r["normalized_squared_errors"].copy(),
            feasible=r["feasible"],
        )
        for r in records
    ]


def test_full_cap_ledger_and_strict_fresh_decisions_pass(saved):
    answer, args = saved
    checked = audit_cap_search(answer, *args)
    assert checked["passed"] and checked["accepted_response_indices"] == [1, 2]
    assert checked["total_nnls_fit_attempts"] == 2
    assert not checked["control_qualified"]
    rollouts, metrics = fresh(answer)
    independent = verify_cap_fresh_decisions(answer, rollouts, metrics, args[-1], diagnostics=[])
    assert independent["passed"] and independent["repeat_exact"]
    assert independent["qualified_witness_indices"] == []


def test_audit_does_not_call_production_projection_or_loss(saved, monkeypatch):
    answer, args = saved

    def forbidden(*args, **kwargs):
        raise AssertionError("Production objective cannot provide independent evidence")

    monkeypatch.setattr(production, "linearize", forbidden)
    monkeypatch.setattr(production, "cap_loss", forbidden)
    assert audit_cap_search(answer, *args)["passed"]
    assert verify_cap_fresh_decisions(answer, *fresh(answer), args[-1])["passed"]


@pytest.mark.parametrize("defect", ["B", "predicted", "actual", "radius", "causal_update"])
def test_corrupt_objective_or_state_machine_is_rejected(saved, defect):
    answer, args = deepcopy(saved)
    stage = answer["stages"][0]
    if defect == "B":
        stage["qp"]["B"][66, 66] += 1
    elif defect == "predicted":
        stage["attempts"][0]["predicted_reduction"] += 1e-3
    elif defect == "actual":
        stage["attempts"][0]["actual_reduction"] += 1e-3
    elif defect == "radius":
        stage["radius_after"] *= 0.5
    else:
        stage["secant_update"]["error_vector"]["reason"] = "invented"
    with pytest.raises(ValueError):
        audit_cap_search(answer, *args)


def test_fresh_pose_mismatch_is_rejected_at_original_tolerance(saved):
    answer, args = saved
    rollouts, metrics = fresh(answer)
    metrics[1]["normalized_squared_errors"][11, 0] += 1e-7
    with pytest.raises(ValueError, match="fresh_actual_reduction"):
        verify_cap_fresh_decisions(answer, rollouts, metrics, args[-1])


def test_restore_uses_original_full_active_certificate():
    evaluate, args = restoration_inputs()
    answer = archive(evaluate, args)
    checked = audit_cap_search(answer, *args)
    row = checked["dual_recovery_checks"][0]
    assert row["mode"] == "restore"
    assert row["recovery_indices"] == row["full_active_indices"]
    assert checked["objective_scaling_checks"][0]["divisor"] == 1


def test_independent_projection_has_full_tangential_derivative_and_strict_zero_set():
    E, q = np.zeros(72), np.zeros(24)
    E[0], q[0] = 2, 4
    residual, derivative = projection_model(E, q, np.eye(72))
    assert_array_equal(residual[:3], [1, 0, 0])
    assert_array_equal(derivative[:3, :3], np.diag([1, 0.5, 0.5]))
    q[0] = np.nextafter(1.0, 2.0)
    assert cap_loss(q) > 0
