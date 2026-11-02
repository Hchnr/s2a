"""Independent algebra checks for saved local QPs and accepted-point curvature.

No production solver, SLSQP, NNLS, model, or trajectory evaluator is imported.
The caller must independently bind B/g/A/b and the actual accepted s/y to the
response ledger. Passing here does not establish nonlinear or control safety.
"""

import numpy as np

TOLERANCE = 1e-6
SCOPE = "Saved local algebra only; no model response, ledger routing, or control qualification"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array(value, name, shape=None, *, allow_nonfinite=False):
    require(not np.iscomplexobj(value), f"{name}: real values required")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name}: numeric values required") from error
    require(shape is None or result.shape == shape, f"{name}: shape mismatch")
    require(allow_nonfinite or np.all(np.isfinite(result)), f"{name}: nonfinite values")
    return result


def spd(value):
    matrix = array(value, "B")
    require(matrix.ndim == 2 and 0 < matrix.shape[0] == matrix.shape[1], "B: nonempty square matrix")
    require(np.array_equal(matrix, matrix.T), "B: exact symmetry")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("B: positive definiteness") from error
    return matrix


def infnorm(value):
    return float(np.max(np.abs(value))) if np.size(value) else 0.0


class Comparisons:
    def __init__(self):
        self.maximum = 0.0

    def check(self, saved, expected, name):
        if expected is None:
            require(saved is None, f"{name}: expected absent diagnostic")
            return
        expected = array(expected, f"expected {name}")
        saved = array(saved, name, expected.shape)
        difference = infnorm(saved - expected)
        self.maximum = max(self.maximum, difference)
        require(np.all(np.abs(saved - expected) <= 1e-12 + 1e-10 * np.abs(expected)), f"{name}: numeric mismatch")


def _qp_incomplete(qp, checks, reason, **known):
    require(qp["qualified"] is False, "Incomplete certificate cannot qualify")
    require(qp["nnls_success"] is False, "Incomplete certificate cannot claim successful dual recovery")
    require(reason in qp["qualification_failures"], "Explicit incomplete-certificate failure reason")
    for name in (
        "multipliers",
        "scaled_multipliers",
        "nnls_residual_norm",
        "stationarity_relative",
        "stationarity_residual_inf",
        "complementarity_relative",
        "complementarity_residual_inf",
        "complementarity_sum_gap",
    ):
        require(qp.get(name) is None, f"{name}: incomplete certificate cannot invent a completed diagnostic")
    return dict(
        passed=True,
        qualified=False,
        certificate_complete=False,
        failure_execution_verified=False,
        failure_reason=reason,
        solver_nit=qp["solver_nit"],
        max_record_difference=checks.maximum,
        scope=SCOPE,
        **known,
    )


def audit_qp_certificate(qp, B, g, A, b):
    """Recompute a saved QP certificate; explicit failed attempts may be partial.

    A partial failure is recognized as unqualified, not independently executed
    or certified. Complete records must exactly reproduce all qualification
    decisions under the registered 1e-6 certificate thresholds.
    """
    B = spd(B)
    n = len(B)
    g, A = array(g, "g", (n,)), array(A, "A")
    require(A.ndim == 2 and A.shape[1] == n + 1, "A: QP variable dimension")
    b = array(b, "b", (len(A),))
    require(type(qp["solver_success"]) is bool and type(qp["qualified"]) is bool, "Explicit QP flags")
    require(type(qp["nnls_success"]) is bool, "Explicit NNLS flag")
    require(type(qp["solver_nit"]) is int and 0 <= qp["solver_nit"] <= 100, "QP iteration budget")
    require(qp["certificate_tolerance"] == TOLERANCE, "Registered certificate tolerance")
    require(np.array_equal(array(qp["B"], "saved B", B.shape), B), "QP input B binding")
    failures = qp["qualification_failures"]
    allowed = {
        "solver_exception",
        "solver_unsuccessful",
        "invalid_solver_point",
        "primal_infeasible",
        "nnls_failed",
        "invalid_certificate",
        "negative_multiplier",
        "stationarity_failed",
        "complementarity_failed",
    }
    require(
        isinstance(failures, list)
        and all(isinstance(item, str) and item in allowed for item in failures)
        and len(set(failures)) == len(failures),
        "Known unique QP failure reasons",
    )
    checks = Comparisons()
    scales = np.maximum.reduce((np.ones(len(A)), np.max(np.abs(A), axis=1), np.abs(b)))
    checks.check(qp["row_scale"], scales, "row_scale")
    if qp["solver_exception"] is not None:
        require(isinstance(qp["solver_exception"], str) and bool(qp["solver_exception"]), "Solver exception text")
        require(
            not qp["solver_success"]
            and qp["solver_status"] is None
            and qp["point"] is None
            and qp["solver_nit"] == 0
            and failures == ["solver_exception"],
            "Solver exception cannot invent a completed point",
        )
        require(
            qp["objective"] is None
            and qp["primal_min"] is None
            and qp["unscaled_primal_min"] is None
            and qp["active_indices"] == []
            and qp["nnls_max_iterations"] is None,
            "Solver exception precedes algebraic certificate",
        )
        return _qp_incomplete(qp, checks, "solver_exception")
    require(type(qp["solver_status"]) is int, "Returned SLSQP status")
    require(qp["solver_success"] == (qp["solver_status"] == 0), "SLSQP status and success agree")
    base_failures = [] if qp["solver_success"] else ["solver_unsuccessful"]
    if qp["point"] is None:
        require(failures == base_failures + ["invalid_solver_point"], "Missing point has explicit failure")
        return _qp_incomplete(qp, checks, "invalid_solver_point")
    point = array(qp["point"], "QP point", (n + 1,))
    d, t = point[:-1], float(point[-1])
    scaled_A, scaled_b = A / scales[:, None], b / scales
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            objective = float(t + np.dot(g, d) + np.dot(d, B @ d) / 2.0)
            grad = np.concatenate((B @ d + g, np.ones(1)))
            raw_slack = A @ point + b
            slack = scaled_A @ point + scaled_b
            require(
                all(np.all(np.isfinite(value)) for value in (objective, grad, raw_slack, slack)), "Finite QP algebra"
            )
    except (FloatingPointError, ValueError):
        require("invalid_certificate" in failures, "Overflow must be an explicit failed certificate")
        return _qp_incomplete(qp, checks, "invalid_certificate")
    active = np.flatnonzero(slack <= 10 * TOLERANCE)
    primal = float(np.min(slack)) if len(slack) else None
    raw_primal = float(np.min(raw_slack)) if len(raw_slack) else None
    known = dict(active_count=len(active), primal_min=primal, unscaled_primal_min=raw_primal)
    scalar_values = dict(objective=objective, primal_min=primal, unscaled_primal_min=raw_primal)
    for name, value in scalar_values.items():
        if qp[name] is not None or "invalid_certificate" not in failures:
            checks.check(qp[name], value, name)
    if "invalid_certificate" in failures and qp["scaled_multipliers"] is None:
        return _qp_incomplete(qp, checks, "invalid_certificate", **known)
    require(qp["active_indices"] == active.tolist(), "Exact scaled active rows")
    require(qp["nnls_max_iterations"] == 3 * max(len(active), 1), "NNLS bounded iteration contract")
    if primal is not None and primal < -TOLERANCE:
        base_failures.append("primal_infeasible")
    if "nnls_failed" in failures:
        require(failures == base_failures + ["nnls_failed"], "NNLS failure stage")
        require(isinstance(qp["nnls_error"], str) and bool(qp["nnls_error"]), "NNLS failure text")
        require(qp["multipliers"] is None and qp["scaled_multipliers"] is None, "No fabricated failed NNLS dual")
        return _qp_incomplete(qp, checks, "nnls_failed", **known)
    require(qp["nnls_success"] and qp["nnls_error"] is None, "Completed NNLS certificate")
    mu = array(qp["scaled_multipliers"], "scaled multipliers", (len(A),))
    dual = array(qp["multipliers"], "original multipliers", (len(A),))
    checks.check(dual, mu / scales, "multipliers row scaling")
    inactive = np.ones(len(A), dtype=bool)
    inactive[active] = False
    require(np.all(mu[inactive] == 0.0), "Inactive rows have exact zero dual")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        dual_force = scaled_A.T @ mu
        residual = grad - dual_force
        stationarity = infnorm(residual)
        products = mu * slack
        complementarity = infnorm(products)
        relative_stationarity = stationarity / max(1.0, infnorm(grad), infnorm(dual_force))
        relative_complementarity = complementarity / max(1.0, abs(objective))
        values = dict(
            stationarity_residual_inf=stationarity,
            stationarity_relative=relative_stationarity,
            complementarity_residual_inf=complementarity,
            complementarity_relative=relative_complementarity,
            complementarity_sum_gap=float(np.sum(products)),
            nnls_residual_norm=float(np.linalg.norm(residual)),
        )
    for name, value in values.items():
        checks.check(qp[name], value, name)
    # Independently check that the original-unit dual represents the same force.
    checks.check(A.T @ dual, dual_force, "original-unit dual force")
    if np.any(mu < 0) or np.any(dual < 0):
        base_failures.append("negative_multiplier")
    if relative_stationarity > TOLERANCE:
        base_failures.append("stationarity_failed")
    if relative_complementarity > TOLERANCE:
        base_failures.append("complementarity_failed")
    require(failures == base_failures, "QP failure decisions agree with independently reconstructed certificate")
    qualified = not base_failures
    require(qp["qualified"] == qualified, "QP qualification agrees with solver status and KKT")
    return dict(
        passed=True,
        qualified=qualified,
        certificate_complete=True,
        solver_nit=qp["solver_nit"],
        stationarity_relative=relative_stationarity,
        complementarity_relative=relative_complementarity,
        max_record_difference=checks.maximum,
        scope=SCOPE,
        **known,
    )


def audit_curvature_update(B, s, y, update):
    """Check Powell's damped secant update at supplied actual accepted s/y.

    The formal contract has minimum_step=0. Nonfinite or numerically invalid
    updates preserve the exact incoming B; no spectral repair is permitted.
    """
    B = spd(B)
    s, y = (
        array(s, "actual s", (len(B),), allow_nonfinite=True),
        array(y, "same-multiplier y", (len(B),), allow_nonfinite=True),
    )
    require(type(update["updated"]) is bool and type(update["damping_applied"]) is bool, "Explicit update flags")
    require(update["minimum_step"] == 0.0, "No additional minimum accepted step")
    checks = Comparisons()
    expected = dict(theta=None, sTy=None, sTBs=None, sTr=None, step_norm=None, gradient_change_norm=None)
    reason, damped, candidate = None, False, None
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
        reason = "nonfinite_step_or_gradient"
    elif np.count_nonzero(s) == 0:
        reason, expected["step_norm"] = "zero_step", 0.0
    else:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                step_norm, gradient_norm = float(np.linalg.norm(s)), float(np.linalg.norm(y))
                if not np.isfinite(step_norm) or not np.isfinite(gradient_norm):
                    raise FloatingPointError("Nonfinite norm")
                expected.update(step_norm=step_norm, gradient_change_norm=gradient_norm)
                if step_norm <= 0:
                    reason = "step_below_minimum"
                else:
                    Bs = B @ s
                    sy, sBs = float(np.dot(s, y)), float(np.dot(s, Bs))
                    if not np.isfinite(sy) or not np.isfinite(sBs):
                        raise FloatingPointError("Nonfinite curvature")
                    expected.update(sTy=sy, sTBs=sBs)
                    if sBs <= 0:
                        reason = "nonpositive_sTBs"
                    else:
                        theta = 1.0 if sy >= 0.2 * sBs else 0.8 * sBs / (sBs - sy)
                        damped = theta != 1.0
                        expected["theta"] = theta
                        r = theta * y + (1.0 - theta) * Bs
                        sr = float(np.dot(s, r))
                        if not np.isfinite(theta) or not np.isfinite(sr) or not np.all(np.isfinite(r)):
                            raise FloatingPointError("Nonfinite damped curvature")
                        expected["sTr"] = sr
                        if theta < 0 or theta > 1 or sr <= 0:
                            reason = "invalid_damped_curvature"
                        else:
                            candidate = B - np.multiply.outer(Bs, Bs) / sBs + np.multiply.outer(r, r) / sr
                            if not np.all(np.isfinite(candidate)) or not np.array_equal(candidate, candidate.T):
                                raise FloatingPointError("Invalid candidate matrix")
                            try:
                                np.linalg.cholesky(candidate)
                            except np.linalg.LinAlgError:
                                reason = "updated_matrix_not_spd"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            reason = "numerical_update_failure"
    changed = reason is None
    reason = "updated" if changed else reason
    require(update["updated"] == changed and update["reason"] == reason, "Curvature update decision")
    require(update["damping_applied"] == damped, "Powell damping decision")
    for name, value in expected.items():
        checks.check(update[name], value, name)
    saved_matrix = spd(update["matrix"])
    if changed:
        checks.check(saved_matrix, candidate, "BFGS matrix")
    else:
        require(np.array_equal(saved_matrix, B), "Failed update must preserve exact previous B")
    return dict(
        passed=True,
        updated=changed,
        reason=reason,
        max_record_difference=checks.maximum,
        scope=SCOPE,
        **expected,
    )
