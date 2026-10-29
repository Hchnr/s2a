"""Stub finite solvers prove whole-objective scaling and original certificates."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_state_vector_gn_search import entry, inputs

import s2a.accepted_point_sqp_core as original_solver
import s2a.state_vector_gn_search as original_search
import s2a.state_vector_objective_scale as planning
from s2a.directional_derivative_probe import _canonical

EXTRA_FIELDS = {
    "solver_objective_divisor",
    "solver_objective_value",
    "objective_scaled_multipliers",
    "objective_scaled_multipliers_provenance",
}


def problem():
    # min t -4d + d^2, d<=1, t>=0: point(1,0), canonical mu(2,1).
    return np.array([[2.0]]), np.array([-4.0]), np.array([[-2.0, 0.0], [0.0, 5.0]]), np.array([2.0, 0.0]), 0.25


def stub_solver(monkeypatch, module, *, success=True, dual=None, error=None, nnls_error=False, point=None):
    calls = dict(minimize=[], nnls=[])
    dual = np.array([2.0, 1.0]) if dual is None else np.asarray(dual)
    point = np.array([1.0, 0.0]) if point is None else np.asarray(point)

    def minimize(objective, initial, *, jac, method, constraints, options):
        probe = np.array([0.4, 0.3])  # Nonzero t catches partial-objective scaling.
        calls["minimize"].append(
            dict(
                initial=initial.copy(),
                objective=objective(probe),
                gradient=jac(probe),
                constraint=constraints[0]["fun"](probe),
                constraint_jacobian=constraints[0]["jac"](probe),
                method=method,
                options=deepcopy(options),
            )
        )
        if error:
            raise RuntimeError("Synthetic finite solver interruption")
        return SimpleNamespace(success=success, status=0 if success else 8, message="synthetic", nit=3, x=point.copy())

    def nnls(matrix, target, *, maxiter):
        calls["nnls"].append(dict(matrix=matrix.copy(), target=target.copy(), maxiter=maxiter))
        if nnls_error:
            raise RuntimeError("Synthetic certificate interruption")
        return dual.copy(), 0.0

    monkeypatch.setattr(module, "minimize", minimize)
    monkeypatch.setattr(module, "nnls", nnls)
    return calls


def test_entire_objective_and_gradient_including_t_are_scaled_while_constraints_and_options_are_unchanged(monkeypatch):
    calls = stub_solver(monkeypatch, planning)
    result = planning.solve_scaled_epigraph_qp(*problem(), objective_divisor=1000)
    finite, certificate = calls["minimize"][0], calls["nnls"][0]
    assert finite["objective"] == pytest.approx((0.3 - 4 * 0.4 + 0.4**2) / 1000)
    np.testing.assert_array_equal(finite["gradient"], np.array([-3.2, 1.0]) / 1000)
    np.testing.assert_array_equal(finite["initial"], [0, 0.25])
    np.testing.assert_array_equal(finite["constraint"], [0.6, 0.3])
    np.testing.assert_array_equal(finite["constraint_jacobian"], [[-1, 0], [0, 1]])
    assert finite["method"] == "SLSQP" and finite["options"] == dict(maxiter=100, ftol=1e-14, disp=False)
    np.testing.assert_array_equal(certificate["target"], [-2, 1])  # Original gradient, including t=1.
    np.testing.assert_array_equal(certificate["matrix"], [[-1, 0], [0, 1]])
    assert certificate["maxiter"] == 6 and len(calls["minimize"]) == len(calls["nnls"]) == 1
    assert result["qualified"] and result["objective"] == -3 and result["solver_objective_value"] == -0.003
    np.testing.assert_array_equal(result["row_scale"], [2, 5])


def test_one_original_nnls_certificate_supplies_both_algebraic_dual_mappings(monkeypatch):
    calls = stub_solver(monkeypatch, planning)
    result = planning.solve_scaled_epigraph_qp(*problem(), objective_divisor=1000)
    mu, nu, S = result["scaled_multipliers"], result["objective_scaled_multipliers"], result["row_scale"]
    np.testing.assert_array_equal(mu, [2, 1])
    np.testing.assert_array_equal(nu, mu / 1000)
    np.testing.assert_allclose(1000 * nu, mu, rtol=1e-15)
    np.testing.assert_allclose(1000 * nu / S, result["multipliers"], rtol=1e-15)
    assert result["objective_scaled_multipliers_provenance"] == "original_nnls_certificate_divided_by_objective_divisor"
    assert result["stationarity_relative"] == result["complementarity_relative"] == 0
    assert len(calls["nnls"]) == 1


def test_saved_solver_objective_includes_nonzero_t_at_a_certificate_admitted_point(monkeypatch):
    stub_solver(monkeypatch, planning, point=[1.0, 5e-7])
    result = planning.solve_scaled_epigraph_qp(*problem(), objective_divisor=1000)
    assert result["qualified"] and result["objective"] == -3 + 5e-7
    assert result["solver_objective_value"] == result["objective"] / 1000
    assert result["solver_objective_value"] != -0.003
    assert 0 < result["complementarity_relative"] < 1e-6


def test_a_small_scaled_kkt_residual_cannot_replace_original_stationarity(monkeypatch):
    calls = stub_solver(monkeypatch, planning, dual=np.array([2, 1]) / 1e9)
    result = planning.solve_scaled_epigraph_qp(*problem(), objective_divisor=1e9)
    assert result["solver_success"] and not result["qualified"]
    assert "stationarity_failed" in result["qualification_failures"]
    assert result["stationarity_relative"] > 0.99
    np.testing.assert_array_equal(calls["nnls"][0]["target"], [-2, 1])


def test_unsuccessful_finite_solver_is_rejected_even_with_an_exact_original_certificate(monkeypatch):
    calls = stub_solver(monkeypatch, planning, success=False)
    result = planning.solve_scaled_epigraph_qp(*problem(), objective_divisor=1000)
    assert result["solver_status"] == 8 and result["stationarity_relative"] == 0
    assert result["qualification_failures"] == ["solver_unsuccessful"] and not result["qualified"]
    assert len(calls["minimize"]) == len(calls["nnls"]) == 1


@pytest.mark.parametrize("failure", [None, "unsuccessful", "solver_exception", "nnls_exception"])
def test_C1_preserves_every_original_solver_field_including_failure_diagnostics(monkeypatch, failure):
    options = dict(
        success=failure != "unsuccessful", error=failure == "solver_exception", nnls_error=failure == "nnls_exception"
    )
    old_calls = stub_solver(monkeypatch, original_solver, **options)
    expected = original_solver.solve_epigraph_qp(*problem())
    new_calls = stub_solver(monkeypatch, planning, **options)
    observed = planning.solve_scaled_epigraph_qp(*problem(), objective_divisor=1.0)
    original_fields = {key: value for key, value in observed.items() if key not in EXTRA_FIELDS}
    assert _canonical(original_fields) == _canonical(expected)
    assert _canonical(old_calls) == _canonical(new_calls)
    assert observed["solver_objective_divisor"] == 1
    assert (observed["solver_objective_value"] is None) == (observed["objective"] is None)
    assert (observed["objective_scaled_multipliers"] is None) == (observed["scaled_multipliers"] is None)


@pytest.mark.parametrize("divisor", [0, -1, np.nan, np.inf, 1j, [1]])
def test_invalid_divisors_precede_the_only_finite_solve(monkeypatch, divisor):
    calls = []
    monkeypatch.setattr(planning, "minimize", lambda *_a, **_k: calls.append(None))
    with pytest.raises(ValueError):
        planning.solve_scaled_epigraph_qp(*problem(), objective_divisor=divisor)
    assert not calls


def test_divisor_uses_matrix_maximum_absolute_row_sum_when_curvature_dominates():
    value = dict(B=np.array([[400.0, -100.0], [-100.0, 200.0]]), g=np.array([0.1, -0.2]))
    assert planning.objective_divisor(value, 0.01, "pose") == 5
    assert planning.objective_divisor(value, 0.01, "pose") != 0.01 * np.max(np.abs(value["B"]))
    value["g"][0] = -9
    assert planning.objective_divisor(value, 0.01, "pose") == 9
    assert planning.objective_divisor(value, 0.01, "restore") == 1
    assert planning.objective_divisor(dict(B=np.eye(2), g=np.zeros(2)), 0.01, "pose") == 1


def fake_search_qp(monkeypatch, module, steps, *, scaled, qualified=True):
    steps, calls = iter(steps), []

    def solve(B, g, A, b, initial_t, **kwargs):
        calls.append(dict(B=B.copy(), g=g.copy(), initial_t=initial_t, kwargs=deepcopy(kwargs)))
        point = np.zeros(73)
        point[0] = next(steps)
        return dict(qualified=qualified, point=point, solver_nit=3)

    monkeypatch.setattr(module, "solve_scaled_epigraph_qp" if scaled else "solve_epigraph_qp", solve)
    return calls


def test_search_forwards_the_fixed_divisor_and_preserves_every_outer_field_with_the_same_point(monkeypatch):
    old_calls = fake_search_qp(monkeypatch, original_search, [0.005], scaled=False)
    expected = original_search.search(**inputs())
    new_calls = fake_search_qp(monkeypatch, planning, [0.005], scaled=True)
    observed = planning.search(**inputs())
    assert observed.pop("qp_objective_scaling_policy") == "pose_whole_objective_divisor"
    assert _canonical(observed) == _canonical(expected)
    assert new_calls[0]["kwargs"] == dict(objective_divisor=15.0, max_iterations=100, certificate_tolerance=1e-6)
    np.testing.assert_array_equal(new_calls[0]["B"], old_calls[0]["B"])
    np.testing.assert_array_equal(new_calls[0]["g"], old_calls[0]["g"])
    assert observed["response_evaluations"] == 3 and observed["selected"] is None


def test_restore_C1_and_subsequent_pose_C_are_bound_from_the_original_stage_problem(monkeypatch):
    value = inputs()
    value["config"]["max_stages"] = 2
    value["Js"][0, 0] = 200e-5
    E = np.zeros(72)
    E[54] = 1.5

    def response(command):
        actual = E.copy()
        actual[54] -= 10 * command[0, 0]
        return entry(command, actual, reserve=0.1 + 200 * command[0, 0])

    value["evaluate"] = response
    value["historical_initial"] = response(np.zeros((12, 6)))
    calls = fake_search_qp(monkeypatch, planning, [0.005, 0.001], scaled=True)
    answer = planning.search(**value)
    assert [stage["mode"] for stage in answer["stages"]] == ["restore", "pose"]
    assert calls[0]["kwargs"]["objective_divisor"] == 1 and calls[0]["initial_t"] > 0
    assert calls[1]["kwargs"]["objective_divisor"] == pytest.approx(
        max(
            1,
            np.max(np.abs(calls[1]["g"])),
            answer["stages"][1]["radius_before"] * np.linalg.norm(calls[1]["B"], ord=np.inf),
        )
    )
    assert calls[1]["initial_t"] == 0 and answer["response_evaluations"] == 4


def test_failed_scaled_qps_keep_the_original_radius_stop_and_never_request_pose_responses(monkeypatch):
    value = inputs()
    value["config"]["max_stages"] = 60
    calls = fake_search_qp(monkeypatch, planning, [0] * 6, scaled=True, qualified=False)
    answer = planning.search(**value)
    assert len(calls) == answer["total_stages"] == 6 and answer["termination_reason"] == "radius_below_minimum"
    assert answer["response_evaluations"] == 2 and answer["unique_command_count"] == 1
    assert answer["selected"] is None and all(not stage["attempts"] for stage in answer["stages"])


def test_response_cap_and_initial_repeat_remain_billed_under_objective_scaling(monkeypatch):
    value = inputs(max_responses=3)
    value["config"]["max_stages"] = 60
    calls = fake_search_qp(monkeypatch, planning, [0.005], scaled=True)
    answer = planning.search(**value)
    assert len(calls) == 1 and answer["termination_reason"] == "response_budget_exhausted"
    assert answer["prediction_calls"] == answer["response_evaluations"] == 3
    assert [row["role"] for row in answer["request_ledger"]] == ["initial", "trial", "initial_repeat"]
    assert answer["repeatability"]["exact"] and answer["final_anchor_index"] == 1
