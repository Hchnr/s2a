"""Fixed tight-slack dual recovery is checked without running a solver or fit."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_state_vector_gn_search import inputs

import s2a.state_vector_dual_recovery as planning
import s2a.state_vector_objective_scale as baseline
from s2a.directional_derivative_probe import _canonical

NEW_FIELDS = {
    "dual_recovery_mode",
    "full_active_threshold",
    "recovery_threshold",
    "recovery_indices",
    "nnls_fit_attempts",
}


def problem(*, redundant=False):
    B, g = np.array([[2.0]]), np.array([-4.0])
    A, b = np.array([[-2.0, 0.0], [0.0, 5.0]]), np.array([2.0, 0.0])
    if redundant:
        A, b = np.array([[-2.0, 0.0], [-2.0, 0.0], [0.0, 5.0]]), np.array([2.0, 2 + 8e-6, 0.0])
    return B, g, A, b, 0.25


def stubs(monkeypatch, module, dual, *, point=(1.0, 0.0), success=True, error=None):
    calls = dict(minimize=[], nnls=[])

    def minimize(objective, initial, *, jac, method, constraints, options):
        probe = np.array([0.4, 0.3])
        calls["minimize"].append(
            dict(
                value=objective(probe),
                gradient=jac(probe),
                initial=initial.copy(),
                method=method,
                constraint=constraints[0]["fun"](probe),
                constraint_jacobian=constraints[0]["jac"](probe),
                options=deepcopy(options),
            )
        )
        if error == "minimize":
            raise RuntimeError("Synthetic primal interruption")
        return SimpleNamespace(
            success=success, status=0 if success else 8, message="synthetic", nit=3, x=np.asarray(point)
        )

    def nnls(matrix, target, *, maxiter):
        calls["nnls"].append(dict(matrix=matrix.copy(), target=target.copy(), maxiter=maxiter))
        if error == "nnls":
            raise RuntimeError("Synthetic NNLS interruption")
        fitted = np.asarray(dual, dtype=float)
        return fitted.copy(), float(np.linalg.norm(matrix @ fitted - target))

    monkeypatch.setattr(module, "minimize", minimize)
    monkeypatch.setattr(module, "nnls", nnls)
    return calls


def test_positive_slack_redundancy_can_fit_stationarity_but_fail_complementarity(monkeypatch):
    arguments = problem(redundant=True)
    positive_row_scale = arguments[3][1]
    old_calls = stubs(monkeypatch, baseline, [0, positive_row_scale, 1])
    old = baseline.solve_scaled_epigraph_qp(*arguments, objective_divisor=1000)
    assert old["solver_success"] and old["stationarity_relative"] < 1e-12
    assert old["complementarity_relative"] > 1e-6 and old["qualification_failures"] == ["complementarity_failed"]
    new_calls = stubs(monkeypatch, planning, [2, 1])
    new = planning.solve_recovered_epigraph_qp(*arguments, objective_divisor=1000)
    assert new["qualified"] and new["stationarity_relative"] == new["complementarity_relative"] == 0
    assert new["active_indices"] == [0, 1, 2] and new["recovery_indices"] == [0, 2]
    assert new["nnls_max_iterations"] == new_calls["nnls"][0]["maxiter"] == 9  # Full set, not the two fitted rows.
    np.testing.assert_array_equal(new["scaled_multipliers"], [2, 0, 1])
    np.testing.assert_array_equal(new["objective_scaled_multipliers"], np.array([2, 0, 1]) / 1000)
    np.testing.assert_array_equal(new_calls["nnls"][0]["target"], [-2, 1])
    np.testing.assert_array_equal(new_calls["nnls"][0]["matrix"], old_calls["nnls"][0]["matrix"][:, [0, 2]])
    assert new["nnls_fit_attempts"] == len(new_calls["nnls"]) == len(new_calls["minimize"]) == 1
    assert _canonical(new_calls["minimize"]) == _canonical(old_calls["minimize"])
    assert new["full_active_threshold"] == 10.0 * 1e-6
    assert new["recovery_threshold"] == 0.01 * 1e-6
    assert (
        new["objective_scaled_multipliers_provenance"]
        == "unscaled_gradient_subset_nnls_certificate_divided_by_objective_divisor"
    )


def test_tight_subset_can_be_insufficient_and_must_not_trigger_a_full_set_fallback(monkeypatch):
    B, g, A, b, initial = problem()
    b[0] += 8e-6  # Only the t row remains tight enough.
    calls = stubs(monkeypatch, planning, [1])
    result = planning.solve_recovered_epigraph_qp(B, g, A, b, initial)
    assert result["active_indices"] == [0, 1] and result["recovery_indices"] == [1]
    np.testing.assert_array_equal(result["scaled_multipliers"], [0, 1])
    assert result["nnls_residual_norm"] == 2 and result["stationarity_relative"] == 1
    assert not result["qualified"] and "stationarity_failed" in result["qualification_failures"]
    assert len(calls["nnls"]) == result["nnls_fit_attempts"] == 1


def test_empty_recovery_has_zero_actual_fits_and_keeps_the_full_constraint_and_budget_checks(monkeypatch):
    B, g, A, b, initial = problem()
    b += np.array([8e-6, 1e-5])
    calls = stubs(monkeypatch, planning, [], error="nnls")
    result = planning.solve_recovered_epigraph_qp(B, g, A, b, initial)
    assert result["active_indices"] == [0, 1] and result["recovery_indices"] == []
    assert result["nnls_max_iterations"] == 6 and result["nnls_fit_attempts"] == len(calls["nnls"]) == 0
    assert result["nnls_residual_norm"] == pytest.approx(np.sqrt(5))
    np.testing.assert_array_equal(result["scaled_multipliers"], [0, 0])
    assert not result["qualified"] and "stationarity_failed" in result["qualification_failures"]


def test_nnls_exception_is_charged_once_with_no_fallback_or_fabricated_dual(monkeypatch):
    calls = stubs(monkeypatch, planning, [2, 1], error="nnls")
    result = planning.solve_recovered_epigraph_qp(*problem(), objective_divisor=10)
    assert result["nnls_fit_attempts"] == len(calls["nnls"]) == 1
    assert not result["nnls_success"] and not result["qualified"]
    assert result["qualification_failures"] == ["nnls_failed"]
    assert result["scaled_multipliers"] is result["multipliers"] is result["objective_scaled_multipliers"] is None
    assert result["solver_objective_divisor"] == 10 and result["solver_objective_value"] == -0.3


def test_original_primal_failure_is_not_hidden_by_a_fitted_tight_set(monkeypatch):
    point = np.array([1.00002, 0.0])
    gradient = 2 * point[0] - 4
    calls = stubs(monkeypatch, planning, [-gradient, 1], point=point)
    result = planning.solve_recovered_epigraph_qp(*problem())
    assert result["stationarity_relative"] == 0 and result["primal_min"] < -1e-6
    assert not result["qualified"] and "primal_infeasible" in result["qualification_failures"]
    assert result["nnls_fit_attempts"] == len(calls["nnls"]) == 1


def test_original_primal_solver_failure_still_blocks_an_exact_dual_certificate(monkeypatch):
    calls = stubs(monkeypatch, planning, [2, 1], success=False)
    result = planning.solve_recovered_epigraph_qp(*problem())
    assert result["solver_status"] == 8 and result["stationarity_relative"] == 0
    assert not result["qualified"] and result["qualification_failures"] == ["solver_unsuccessful"]
    assert result["nnls_fit_attempts"] == len(calls["nnls"]) == 1


@pytest.mark.parametrize("failure", ["minimize", "invalid_point"])
def test_primal_exception_or_invalid_point_never_claims_a_dual_fit(monkeypatch, failure):
    calls = stubs(monkeypatch, planning, [2, 1], error=failure, point=[1] if failure == "invalid_point" else [1, 0])
    result = planning.solve_recovered_epigraph_qp(*problem())
    assert not result["qualified"] and result["nnls_fit_attempts"] == len(calls["nnls"]) == 0
    assert result["recovery_indices"] == result["active_indices"] == []
    assert result["objective_scaled_multipliers"] is result["scaled_multipliers"] is None
    assert result["dual_recovery_mode"] == "pose"


@pytest.mark.parametrize("redundant", [False, True])
def test_restore_preserves_all_previous_fields_and_its_full_active_fit_exactly(monkeypatch, redundant):
    arguments = problem(redundant=redundant)
    dual = [0, arguments[3][1], 1] if redundant else [2, 1]
    old_calls = stubs(monkeypatch, baseline, dual)
    expected = baseline.solve_scaled_epigraph_qp(*arguments, objective_divisor=1)
    new_calls = stubs(monkeypatch, planning, dual)
    observed = planning.solve_recovered_epigraph_qp(*arguments, objective_divisor=1, dual_recovery_mode="restore")
    assert _canonical({key: value for key, value in observed.items() if key not in NEW_FIELDS}) == _canonical(expected)
    assert _canonical(old_calls) == _canonical(new_calls)
    assert observed["recovery_indices"] == observed["active_indices"]
    assert observed["recovery_threshold"] == observed["full_active_threshold"]
    assert (
        observed["objective_scaled_multipliers_provenance"] == "original_nnls_certificate_divided_by_objective_divisor"
    )


def test_bad_mode_is_rejected_before_any_primal_or_fit_call(monkeypatch):
    calls = stubs(monkeypatch, planning, [2, 1])
    with pytest.raises(ValueError):
        planning.solve_recovered_epigraph_qp(*problem(), dual_recovery_mode="full_then_retry")
    assert not calls["minimize"] and not calls["nnls"]


def fake_search_qp(monkeypatch, module, steps, *, recovered, qualified=True):
    calls, steps = [], iter(steps)

    def solve(B, g, A, b, initial_t, **kwargs):
        calls.append(dict(B=B.copy(), g=g.copy(), A=A.copy(), b=b.copy(), initial_t=initial_t, kwargs=deepcopy(kwargs)))
        point = np.zeros(73)
        point[0] = next(steps)
        return dict(qualified=qualified, point=point, solver_nit=3)

    monkeypatch.setattr(module, "solve_recovered_epigraph_qp" if recovered else "solve_scaled_epigraph_qp", solve)
    return calls


def test_search_only_forwards_the_mode_and_keeps_every_outer_gate_and_field(monkeypatch):
    fake_search_qp(monkeypatch, baseline, [0.005], recovered=False)
    expected = baseline.search(**inputs())
    calls = fake_search_qp(monkeypatch, planning, [0.005], recovered=True)
    observed = planning.search(**inputs())
    assert observed.pop("dual_recovery_policy") == "pose_slack_restricted_nnls_restore_full_active"
    assert _canonical(expected) == _canonical(observed)
    assert calls[0]["kwargs"] == dict(
        objective_divisor=15.0, dual_recovery_mode="pose", max_iterations=100, certificate_tolerance=1e-6
    )
    assert observed["response_evaluations"] == 3 and observed["selected"] is None


def test_search_restoration_uses_the_original_full_fit_mode_before_pose_subset_mode(monkeypatch):
    value = inputs()
    value["config"]["max_stages"] = 2
    value["Js"][0, 0] = 200e-5
    original = value["evaluate"]

    def response(command):
        result = original(command)
        result["safety_slack"][0] = 1e-8 + (0.1 + 200 * command[0, 0]) * 1e-5
        return result

    value["evaluate"] = response
    value["historical_initial"] = response(np.zeros((12, 6)))
    calls = fake_search_qp(monkeypatch, planning, [0.005, 0.001], recovered=True)
    answer = planning.search(**value)
    assert [stage["mode"] for stage in answer["stages"]] == ["restore", "pose"]
    assert [call["kwargs"]["dual_recovery_mode"] for call in calls] == ["restore", "pose"]
    assert calls[0]["kwargs"]["objective_divisor"] == 1 and answer["response_evaluations"] == 4


def test_unqualified_recovered_qp_never_requests_a_response_or_restarts_its_budget(monkeypatch):
    value = inputs()
    value["config"]["max_stages"] = 60
    calls = fake_search_qp(monkeypatch, planning, [0] * 6, recovered=True, qualified=False)
    answer = planning.search(**value)
    assert len(calls) == answer["total_stages"] == 6
    assert answer["termination_reason"] == "radius_below_minimum" and answer["response_evaluations"] == 2
    assert answer["selected"] is None and all(not stage["attempts"] for stage in answer["stages"])


def test_original_response_limit_and_charged_initial_repeat_remain_unchanged(monkeypatch):
    value = inputs(max_responses=3)
    value["config"]["max_stages"] = 60
    calls = fake_search_qp(monkeypatch, planning, [0.005], recovered=True)
    answer = planning.search(**value)
    assert len(calls) == 1 and answer["termination_reason"] == "response_budget_exhausted"
    assert answer["prediction_calls"] == answer["response_evaluations"] == 3 and answer["unique_command_count"] == 2
    assert [row["role"] for row in answer["request_ledger"]] == ["initial", "trial", "initial_repeat"]
    assert answer["repeatability"]["exact"]
