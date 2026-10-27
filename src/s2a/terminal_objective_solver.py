"""Matrix-free ridge quadratic and smoothed-LAD solvers with explicit budgets."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def ridge_quadratic(operator, offset, ridge, *, maxiter=1000, rtol=1e-9, initial=None):
    """Minimize ||A d+b||² + ridge||d||²; return actual residual evidence."""
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Positive ridge required")
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (operator.shape[0],) or not np.isfinite(offset).all():
        raise ValueError("Finite operator-sized offset required")
    rhs = -operator.rmatvec(offset)
    normal = LinearOperator(
        (operator.shape[1], operator.shape[1]),
        matvec=lambda d: operator.rmatvec(operator.matvec(d)) + ridge * d,
        dtype=np.float64,
    )
    iterations = 0

    def count(_):
        nonlocal iterations
        iterations += 1

    result, info = cg(normal, rhs, x0=initial, rtol=rtol, atol=0, maxiter=maxiter, callback=count)
    residual = normal.matvec(result) - rhs
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny))
    return result, dict(cg_info=int(info), iterations=iterations, relative_normal_residual=relative)


def lad_certificate(operator, offset, weights, delta, ridge, epsilon):
    """Feasible dual bound for original weighted L1 + ridge, not smoothed loss.

    L(d,s)=sᵀ(b+Ad)+ridge||d||², |s|<=weights, hence
    dual(s)=sᵀb-||Aᵀs||²/(4*ridge). This remains valid off optimum.
    """
    error = operator.matvec(delta) + offset
    dual = weights * error / np.sqrt(error**2 + epsilon**2)
    adjoint = operator.rmatvec(dual)
    primal = float(weights @ abs(error) + ridge * (delta @ delta))
    lower = float(dual @ offset - (adjoint @ adjoint) / (4 * ridge))
    return dict(primal=primal, dual=lower, gap=primal - lower, feasible=bool((abs(dual) <= weights).all()))


def fit_opening_lad(operator, offset, weights, ridge, *, epsilon=1e-4, outer_steps=30, maxiter=1000):
    """Fixed-budget IRLS; no retry or epsilon schedule, original-L1 certificate."""
    offset, weights = np.asarray(offset, dtype=float), np.asarray(weights, dtype=float)
    if (
        offset.shape != (operator.shape[0],)
        or weights.shape != offset.shape
        or not np.isfinite(offset).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("Finite offsets, positive-sum weights and positive epsilon required")
    delta, history = np.zeros(operator.shape[1]), []
    for iteration in range(outer_steps):
        error = operator.matvec(delta) + offset
        # sqrt(e²+epsilon²) majorizer has quadratic coefficient 1/(2 sqrt(...)).
        root_weight = np.sqrt(weights / (2 * np.sqrt(error**2 + epsilon**2)))
        weighted = LinearOperator(
            operator.shape,
            matvec=lambda d: root_weight * operator.matvec(d),
            rmatvec=lambda v: operator.rmatvec(root_weight * v),
            dtype=np.float64,
        )
        delta, evidence = ridge_quadratic(
            weighted,
            root_weight * offset,
            ridge,
            initial=delta,
            maxiter=maxiter,
        )
        certificate = lad_certificate(operator, offset, weights, delta, ridge, epsilon)
        history.append(dict(outer_step=iteration + 1, **evidence, **certificate))
        if evidence["cg_info"] != 0:
            break
    return delta, history
