"""Feasible active-set QP candidate with null-space directions and explicit KKT checks."""

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import nnls


def solve_qp(hessian, linear, constraints, bounds, initial):
    h, g, a, b, x = (np.asarray(v, dtype=float).copy() for v in (hessian, linear, constraints, bounds, initial))
    n = len(x)
    if h.shape != (n, n) or g.shape != (n,) or a.ndim != 2 or a.shape[1] != n or b.shape != (len(a),):
        raise ValueError("Aligned QP inputs required")
    if not all(np.isfinite(v).all() for v in (h, g, a, b, x)) or not np.allclose(h, h.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Finite symmetric QP required")
    np.linalg.cholesky(h)
    if np.max(a @ x - b) > 1e-10:
        raise ValueError("Feasible initial point required")
    working = []
    for i in np.flatnonzero(b - a @ x <= 1e-10):
        if np.linalg.matrix_rank(a[working + [i]], tol=1e-10) > len(working):
            working.append(int(i))
    for iteration in range(1000):
        aw = a[working]
        gradient = h @ x + g
        z = null_space(aw, rcond=1e-12) if working else np.eye(n)
        direction = -z @ np.linalg.solve(z.T @ h @ z, z.T @ gradient) if z.shape[1] else np.zeros(n)
        if working:
            direction -= aw.T @ np.linalg.solve(aw @ aw.T, aw @ direction)
        if np.max(np.abs(direction)) <= 1e-10:
            multipliers = np.linalg.lstsq(aw.T, -gradient, rcond=None)[0]
            if working and multipliers.min() < -1e-9:
                working.pop(int(np.argmin(multipliers)))
                continue
            slack = b - a @ x
            active = np.flatnonzero(slack <= 1e-8)
            dual = nnls(a[active].T, -gradient, maxiter=10000)[0] if len(active) else np.zeros(0)
            stationarity = gradient + a[active].T @ dual
            complementarity = float(np.max(np.abs(dual * slack[active]))) if len(active) else 0.0
            if slack.min() < -1e-9 or np.max(np.abs(stationarity)) > 1e-8 or complementarity > 1e-8:
                raise RuntimeError("Null-space QP fails unchanged primal/KKT gate")
            return dict(
                solution=x,
                iterations=iteration + 1,
                active=active.tolist(),
                multipliers=dual,
                max_stationarity=float(np.max(np.abs(stationarity))),
                max_complementarity=complementarity,
            )
        slack, change = b - a @ x, a @ direction
        fraction, blocking = 1.0, None
        for i in range(len(a)):
            if i in working or change[i] <= 1e-12:
                continue
            # A dependent normal cannot block a direction tangent to the working
            # face. Ignore it only when its current slack is also feasible.
            if working and np.linalg.matrix_rank(a[working + [i]], tol=1e-10) == len(working):
                if slack[i] < -1e-9:
                    raise RuntimeError("Dependent inequality is infeasible")
                continue
            candidate = max(0.0, float(slack[i] / change[i]))
            if candidate < fraction:
                fraction, blocking = candidate, i
        x += fraction * direction
        if np.max(a @ x - b) > 1e-9:
            raise RuntimeError("Null-space step left feasible region")
        if blocking is not None and fraction < 1 - 1e-12:
            working.append(blocking)
    raise RuntimeError("Null-space QP iteration budget exhausted")
