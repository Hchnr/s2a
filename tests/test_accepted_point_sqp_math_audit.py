"""Analytic archived QPs/secants and meaningful certificate tampering checks."""

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from s2a.accepted_point_sqp_core import damped_bfgs, solve_epigraph_qp


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("accepted_point_sqp_math_audit")


def analytic_archive():
    # min t - d/2 + d^2, t >= 1-d, t >= 1+d, t>=0.
    # The exact optimum (0,1) has duals (1/4,3/4,0) before row rescaling.
    # Unequal binary scales distinguish original lambda from scaled mu.
    B, g = np.array([[2.0]]), np.array([-0.5])
    factors = np.array([8.0, 0.25, 3.0])
    A = factors[:, None] * np.array([[1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]])
    b = factors * np.array([-1.0, -1.0, 0.0])
    qp = dict(
        B=B.copy(),
        point=np.array([0.0, 1.0]),
        solver_success=True,
        solver_status=0,
        solver_message="Analytic fixture, not an executed solver",
        solver_nit=1,
        solver_exception=None,
        qualified=True,
        objective=1.0,
        multipliers=np.array([0.03125, 3.0, 0.0]),
        scaled_multipliers=np.array([0.25, 3.0, 0.0]),
        row_scale=np.array([8.0, 1.0, 3.0]),
        active_indices=[0, 1],
        primal_min=0.0,
        unscaled_primal_min=0.0,
        stationarity_relative=0.0,
        stationarity_residual_inf=0.0,
        complementarity_relative=0.0,
        complementarity_residual_inf=0.0,
        complementarity_sum_gap=0.0,
        certificate_tolerance=1e-6,
        nnls_success=True,
        nnls_error=None,
        nnls_residual_norm=0.0,
        nnls_max_iterations=6,
        qualification_failures=[],
    )
    return qp, B, g, A, b


def test_known_epigraph_optimum_and_dual_units(audit):
    qp, B, g, A, b = analytic_archive()
    np.testing.assert_array_equal(A.T @ qp["multipliers"], [-0.5, 1.0])
    np.testing.assert_array_equal(qp["multipliers"] * (A @ qp["point"] + b), 0.0)
    result = audit.audit_qp_certificate(qp, B, g, A, b)
    assert result["qualified"] and result["certificate_complete"]
    assert result["active_count"] == 2 and result["max_record_difference"] == 0


def test_actual_public_solver_archive_has_same_analytic_solution(audit):
    _, B, g, A, b = analytic_archive()
    qp = solve_epigraph_qp(B, g, A, b, 2.0)
    np.testing.assert_allclose(qp["point"], [0, 1], rtol=0, atol=1e-8)
    assert audit.audit_qp_certificate(qp, B, g, A, b)["qualified"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective", 0.5),
        ("stationarity_relative", 0.01),
        ("complementarity_relative", 0.01),
        ("complementarity_sum_gap", 0.01),
        ("primal_min", -0.1),
        ("active_indices", [0]),
        ("solver_nit", 101),
        ("certificate_tolerance", 0.01),
        ("nnls_max_iterations", 100),
        ("qualified", False),
    ],
)
def test_qp_tampered_certificate_rejected(audit, field, value):
    qp, *arguments = analytic_archive()
    qp[field] = value
    with pytest.raises(ValueError):
        audit.audit_qp_certificate(qp, *arguments)


@pytest.mark.parametrize("mode", ["raw_dual", "row_scale", "both_duals", "inactive_dual", "negative_dual"])
def test_dual_tampering_cannot_hide_behind_saved_residuals(audit, mode):
    qp, *arguments = analytic_archive()
    if mode == "raw_dual":
        qp["multipliers"][0] *= 8
    elif mode == "row_scale":
        qp["row_scale"][0] = 1
    elif mode == "both_duals":
        qp["multipliers"][0] *= 2
        qp["scaled_multipliers"][0] *= 2
    elif mode == "inactive_dual":
        qp["multipliers"][2] = 0.1
        qp["scaled_multipliers"][2] = 0.3
    else:
        qp["multipliers"][0] *= -1
        qp["scaled_multipliers"][0] *= -1
    with pytest.raises(ValueError):
        audit.audit_qp_certificate(qp, *arguments)


def test_solver_failure_cannot_be_relabelled_as_qualified(audit):
    qp, *arguments = analytic_archive()
    qp.update(solver_success=False, solver_status=9, qualified=False, qualification_failures=["solver_unsuccessful"])
    result = audit.audit_qp_certificate(qp, *arguments)
    assert result["certificate_complete"] and not result["qualified"]
    qp["qualified"] = True
    with pytest.raises(ValueError):
        audit.audit_qp_certificate(qp, *arguments)


def test_nnls_failure_is_only_a_partial_unqualified_record(audit):
    qp, *arguments = analytic_archive()
    qp.update(
        qualified=False,
        nnls_success=False,
        nnls_error="RuntimeError: finite NNLS iteration budget",
        qualification_failures=["nnls_failed"],
        multipliers=None,
        scaled_multipliers=None,
        nnls_residual_norm=None,
    )
    for name in (
        "stationarity_relative",
        "stationarity_residual_inf",
        "complementarity_relative",
        "complementarity_residual_inf",
        "complementarity_sum_gap",
    ):
        qp[name] = None
    result = audit.audit_qp_certificate(qp, *arguments)
    assert not result["qualified"] and not result["certificate_complete"]
    assert not result["failure_execution_verified"]
    qp["qualified"] = True
    with pytest.raises(ValueError):
        audit.audit_qp_certificate(qp, *arguments)


def test_solver_exception_does_not_require_invented_math(audit):
    qp, *arguments = analytic_archive()
    qp.update(
        solver_success=False,
        solver_status=None,
        solver_nit=0,
        solver_exception="FloatingPointError: overflow",
        point=None,
        qualified=False,
        nnls_success=False,
        qualification_failures=["solver_exception"],
        active_indices=[],
    )
    for name in (
        "objective",
        "primal_min",
        "unscaled_primal_min",
        "nnls_max_iterations",
        "multipliers",
        "scaled_multipliers",
        "nnls_residual_norm",
        "stationarity_relative",
        "stationarity_residual_inf",
        "complementarity_relative",
        "complementarity_residual_inf",
        "complementarity_sum_gap",
    ):
        qp[name] = None
    result = audit.audit_qp_certificate(qp, *arguments)
    assert result["passed"] and not result["qualified"]


def negative_secant_archive():
    # B=I, s=e1, y=-e1 => theta=.4, r=.2e1, B_new=diag(.2,1).
    B, s, y = np.eye(2), np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    update = dict(
        matrix=np.diag([0.2, 1.0]),
        updated=True,
        reason="updated",
        theta=0.4,
        sTy=-1.0,
        sTBs=1.0,
        sTr=0.2,
        step_norm=1.0,
        gradient_change_norm=1.0,
        minimum_step=0.0,
        damping_applied=True,
    )
    return B, s, y, update


def test_negative_curvature_uses_known_powell_secant(audit):
    B, s, y, update = negative_secant_archive()
    np.testing.assert_allclose(update["matrix"] @ s, 0.2 * s)
    result = audit.audit_curvature_update(B, s, y, update)
    assert result["updated"] and result["theta"] == 0.4
    produced = damped_bfgs(B, s, y)
    assert audit.audit_curvature_update(B, s, y, produced)["updated"]


@pytest.mark.parametrize("mode", ["actual_s", "actual_y", "theta", "sTy", "matrix", "reset", "minimum_step"])
def test_curvature_tampering_or_wrong_accepted_point_rejected(audit, mode):
    B, s, y, update = negative_secant_archive()
    if mode == "actual_s":
        s *= 0.5
    elif mode == "actual_y":
        y *= -1
    elif mode == "theta":
        update["theta"] = 1.0
    elif mode == "sTy":
        update["sTy"] = 1.0
    elif mode == "matrix":
        update["matrix"][1, 1] = 2
    elif mode == "reset":
        update.update(updated=False, reason="zero_step", matrix=B.copy())
    else:
        update["minimum_step"] = 1e-4
    with pytest.raises(ValueError):
        audit.audit_curvature_update(B, s, y, update)


def test_two_small_accepted_secants_preserve_observed_curvature(audit):
    B = np.eye(2)
    for axis, curvature in enumerate([2.0, 3.0]):
        s = np.zeros(2)
        s[axis] = 1e-8
        y = curvature * s
        update = damped_bfgs(B, s, y)
        assert audit.audit_curvature_update(B, s, y, update)["updated"]
        B = update["matrix"]
    np.testing.assert_allclose(B, np.diag([2.0, 3.0]), atol=1e-12)


@pytest.mark.parametrize("s,y", [(np.zeros(2), np.ones(2)), (np.ones(2), np.array([np.nan, 1.0]))])
def test_invalid_update_retains_exact_previous_matrix(audit, s, y):
    B = np.array([[2.0, 0.25], [0.25, 1.0]])
    update = damped_bfgs(B, s, y)
    result = audit.audit_curvature_update(B, s, y, update)
    assert not result["updated"]
    damaged = deepcopy(update)
    damaged["matrix"] = np.eye(2)
    with pytest.raises(ValueError):
        audit.audit_curvature_update(B, s, y, damaged)


def test_audit_has_no_optimizer_or_production_import(audit):
    module = ast.parse(Path(audit.__file__).read_text())
    imports = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert imports == ["numpy"]
