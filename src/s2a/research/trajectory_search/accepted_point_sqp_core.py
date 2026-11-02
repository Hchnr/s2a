"""Algebraic QP certificates and curvature updates at accepted SQP points.

These routines never evaluate a trajectory. A QP certificate concerns only
the supplied local quadratic model, not the nonlinear constraints or control.
"""

import numpy as np
from scipy.optimize import minimize, nnls


def _array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        return np.asarray(value, dtype=np.float64).copy()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real numeric array") from error


def _spd(value):
    matrix = _array(value, "B")
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or not np.array_equal(matrix, matrix.T)
    ):
        raise ValueError("B must be a finite symmetric nonempty square matrix")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("B must be positive definite") from error
    return matrix


def _scalar(value, name):
    result = _array(value, name)
    if result.shape != () or not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return float(result)


def _inf(value):
    return float(np.max(np.abs(value))) if np.size(value) else 0.0


def solve_epigraph_qp(B, g, A, b, initial_t, *, max_iterations=100, certificate_tolerance=1e-6):
    """Solve and certify ``min t + g.d + .5 d.B.d``, with ``A[d,t]+b>=0``.

    All rows, including command bounds and ``t>=0``, belong to the caller.
    Rows are divided by ``max(1, max(abs(A_row)), abs(b_row))``. The primal
    certificate uses these scaled slacks. Multipliers are nonnegative for the
    convention ``L = objective - multipliers @ constraints``.

    Both SLSQP success and the independent NNLS KKT certificate are necessary
    for qualification. Solver and certificate failures retain diagnostics;
    they do not trigger retries or supply a qualified direction.
    """
    B = _spd(B)
    width = len(B)
    g, A, b = _array(g, "g"), _array(A, "A"), _array(b, "b")
    initial_t = _scalar(initial_t, "initial_t")
    tolerance = _scalar(certificate_tolerance, "certificate_tolerance")
    if (
        g.shape != (width,)
        or A.ndim != 2
        or A.shape[1] != width + 1
        or b.shape != (len(A),)
        or not all(np.all(np.isfinite(value)) for value in (g, A, b))
        or type(max_iterations) is not int
        or max_iterations < 1
        or tolerance <= 0.0
    ):
        raise ValueError("Finite consistent QP arrays and positive explicit budgets/tolerance required")
    row_scale = np.maximum(1.0, np.maximum(np.max(np.abs(A), axis=1), np.abs(b)))
    scaled_A, scaled_b = A / row_scale[:, None], b / row_scale
    initial = np.r_[np.zeros(width), initial_t]
    result = dict(
        B=B.copy(),
        point=None,
        solver_success=False,
        solver_status=None,
        solver_message=None,
        solver_nit=0,
        solver_exception=None,
        qualified=False,
        objective=None,
        multipliers=None,
        scaled_multipliers=None,
        row_scale=row_scale.copy(),
        active_indices=[],
        primal_min=None,
        unscaled_primal_min=None,
        stationarity_relative=None,
        stationarity_residual_inf=None,
        complementarity_relative=None,
        complementarity_residual_inf=None,
        complementarity_sum_gap=None,
        certificate_tolerance=tolerance,
        nnls_success=False,
        nnls_error=None,
        nnls_residual_norm=None,
        nnls_max_iterations=None,
        qualification_failures=[],
    )

    def objective(point):
        with np.errstate(over="raise", invalid="raise"):
            d = point[:width]
            return float(point[width] + g @ d + 0.5 * (d @ (B @ d)))

    def gradient(point):
        with np.errstate(over="raise", invalid="raise"):
            return np.r_[g + B @ point[:width], 1.0]

    def constraint(point):
        with np.errstate(over="raise", invalid="raise"):
            return scaled_A @ point + scaled_b

    try:
        solved = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            constraints=[dict(type="ineq", fun=constraint, jac=lambda point: scaled_A.copy())],
            options=dict(maxiter=max_iterations, ftol=1e-14, disp=False),
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        result.update(
            solver_message=str(error),
            solver_exception=f"{type(error).__name__}: {error}",
            qualification_failures=["solver_exception"],
        )
        return result
    result.update(
        solver_success=bool(solved.success),
        solver_status=int(solved.status),
        solver_message=str(solved.message),
        solver_nit=int(solved.nit),
    )
    if not result["solver_success"]:
        result["qualification_failures"].append("solver_unsuccessful")
    point = _array(solved.x, "solver point")
    if point.shape != (width + 1,) or not np.all(np.isfinite(point)):
        result["qualification_failures"].append("invalid_solver_point")
        return result
    result["point"] = point.copy()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value, grad, slack = objective(point), gradient(point), constraint(point)
            raw_slack = A @ point + b
            if not all(np.all(np.isfinite(item)) for item in (value, grad, slack, raw_slack)):
                raise FloatingPointError("Nonfinite QP certificate inputs")
            active = np.flatnonzero(slack <= 10.0 * tolerance)
            result.update(
                objective=value,
                active_indices=active.tolist(),
                primal_min=float(np.min(slack)) if len(slack) else None,
                unscaled_primal_min=float(np.min(raw_slack)) if len(raw_slack) else None,
                nnls_max_iterations=3 * max(len(active), 1),
            )
            if len(slack) and result["primal_min"] < -tolerance:
                result["qualification_failures"].append("primal_infeasible")
            mu = np.zeros(len(A))
            try:
                if len(active):
                    fitted, residual = nnls(scaled_A[active].T, grad, maxiter=result["nnls_max_iterations"])
                    if fitted.shape != (len(active),) or not np.all(np.isfinite(fitted)):
                        raise ValueError("NNLS returned invalid multipliers")
                    mu[active] = fitted
                else:
                    residual = float(np.linalg.norm(grad))
                if not np.isfinite(residual):
                    raise ValueError("NNLS returned a nonfinite residual")
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                result.update(nnls_error=f"{type(error).__name__}: {error}")
                result["qualification_failures"].append("nnls_failed")
                return result
            multipliers = mu / row_scale
            dual_force = scaled_A.T @ mu
            stationarity = _inf(grad - dual_force)
            products = mu * slack
            complementarity = _inf(products)
            relative_stationarity = stationarity / max(1.0, _inf(grad), _inf(dual_force))
            relative_complementarity = complementarity / max(1.0, abs(value))
            gap = float(np.sum(products))
            if not all(
                np.all(np.isfinite(item))
                for item in (multipliers, dual_force, products, relative_stationarity, relative_complementarity, gap)
            ):
                raise FloatingPointError("Nonfinite QP certificate")
            result.update(
                multipliers=multipliers,
                scaled_multipliers=mu,
                nnls_success=True,
                nnls_residual_norm=float(residual),
                stationarity_residual_inf=stationarity,
                stationarity_relative=relative_stationarity,
                complementarity_residual_inf=complementarity,
                complementarity_relative=relative_complementarity,
                complementarity_sum_gap=gap,
            )
            if np.any(mu < 0.0) or np.any(multipliers < 0.0):
                result["qualification_failures"].append("negative_multiplier")
            if relative_stationarity > tolerance:
                result["qualification_failures"].append("stationarity_failed")
            if relative_complementarity > tolerance:
                result["qualification_failures"].append("complementarity_failed")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        result["certificate_error"] = f"{type(error).__name__}: {error}"
        result["qualification_failures"].append("invalid_certificate")
        return result
    result["qualified"] = not result["qualification_failures"]
    return result


def damped_bfgs(B, s, y, *, minimum_step=0.0):
    """Apply Powell damping (eta=.2) using an actual accepted displacement.

    The caller supplies the same-multiplier Lagrangian gradient difference.
    Failed updates retain the exact prior matrix: no reset, eigenvalue repair,
    or rejected-point curvature is introduced here. Step size uses L2 norm.
    """
    B = _spd(B)
    s, y = _array(s, "s"), _array(y, "y")
    minimum_step = _scalar(minimum_step, "minimum_step")
    if s.shape != (len(B),) or y.shape != (len(B),) or minimum_step < 0.0:
        raise ValueError("Matching displacement/gradient vectors and nonnegative minimum_step required")
    result = dict(
        matrix=B.copy(),
        updated=False,
        reason=None,
        theta=None,
        sTy=None,
        sTBs=None,
        sTr=None,
        step_norm=None,
        gradient_change_norm=None,
        minimum_step=minimum_step,
        damping_applied=False,
    )
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)):
        result["reason"] = "nonfinite_step_or_gradient"
        return result
    if not np.any(s != 0.0):
        result.update(reason="zero_step", step_norm=0.0)
        return result
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            step_norm, gradient_norm = float(np.linalg.norm(s)), float(np.linalg.norm(y))
            if not np.isfinite(step_norm) or not np.isfinite(gradient_norm):
                raise FloatingPointError("Nonfinite vector norm")
            result.update(step_norm=step_norm, gradient_change_norm=gradient_norm)
            if step_norm <= minimum_step:
                result["reason"] = "step_below_minimum"
                return result
            Bs = B @ s
            sTy, sTBs = float(s @ y), float(s @ Bs)
            if not np.isfinite(sTy) or not np.isfinite(sTBs):
                raise FloatingPointError("Nonfinite curvature products")
            result.update(sTy=sTy, sTBs=sTBs)
            if sTBs <= 0.0:
                result["reason"] = "nonpositive_sTBs"
                return result
            theta = 1.0 if sTy >= 0.2 * sTBs else 0.8 * sTBs / (sTBs - sTy)
            result.update(theta=float(theta), damping_applied=theta != 1.0)
            r = theta * y + (1.0 - theta) * Bs
            sTr = float(s @ r)
            if not np.isfinite(theta) or not np.isfinite(sTr) or not np.all(np.isfinite(r)):
                raise FloatingPointError("Nonfinite damped curvature")
            result["sTr"] = sTr
            if not 0.0 <= theta <= 1.0 or sTr <= 0.0:
                result["reason"] = "invalid_damped_curvature"
                return result
            candidate = B - np.outer(Bs, Bs) / sTBs + np.outer(r, r) / sTr
            if not np.all(np.isfinite(candidate)) or not np.array_equal(candidate, candidate.T):
                raise FloatingPointError("Nonfinite or nonsymmetric BFGS matrix")
            try:
                np.linalg.cholesky(candidate)
            except np.linalg.LinAlgError:
                result["reason"] = "updated_matrix_not_spd"
                return result
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        result.update(reason="numerical_update_failure", error=f"{type(error).__name__}: {error}")
        return result
    result.update(matrix=candidate, updated=True, reason="updated")
    return result
