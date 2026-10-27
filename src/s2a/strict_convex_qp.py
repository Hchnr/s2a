"""219 feasible primal active set for small strictly convex quadratic programs."""

import numpy as np


def solve_qp(hessian, linear, constraints, bounds, initial):
    hessian, linear, a, b, x = (
        np.asarray(v, dtype=float).copy() for v in (hessian, linear, constraints, bounds, initial)
    )
    n = len(x)
    if (
        hessian.shape != (n, n)
        or linear.shape != (n,)
        or a.ndim != 2
        or a.shape[1] != n
        or b.shape != (len(a),)
        or not all(np.isfinite(v).all() for v in (hessian, linear, a, b, x))
    ):
        raise ValueError("Finite aligned strictly convex QP required")
    np.linalg.cholesky(hessian)
    if np.max(a @ x - b) > 1e-10:
        raise ValueError("Feasible initial actual-command sequence required")
    working = []
    for index in np.flatnonzero(b - a @ x <= 1e-10):
        trial = working + [int(index)]
        if np.linalg.matrix_rank(a[trial], tol=1e-10) == len(trial):
            working = trial
    for iteration in range(1000):
        gradient = hessian @ x + linear
        aw = a[working]
        kkt = np.block([[hessian, aw.T], [aw, np.zeros((len(working), len(working)))]])
        solution = np.linalg.solve(kkt, np.r_[-gradient, np.zeros(len(working))])
        direction, multipliers = solution[:n], solution[n:]
        if np.max(np.abs(direction)) <= 1e-10:
            if len(working) and np.min(multipliers) < -1e-9:
                working.pop(int(np.argmin(multipliers)))
                continue
            stationarity = gradient + aw.T @ multipliers
            if np.max(a @ x - b) > 1e-9 or np.max(np.abs(stationarity)) > 1e-8:
                raise RuntimeError("Active-set QP failed final primal/stationarity check")
            return dict(
                solution=x,
                iterations=iteration + 1,
                active=working,
                multipliers=multipliers,
                max_stationarity=float(np.max(np.abs(stationarity))),
            )
        fraction, blocking = 1.0, None
        slack, change = b - a @ x, a @ direction
        for index in range(len(a)):
            if index in working or change[index] <= 1e-12:
                continue
            candidate = max(0.0, float(slack[index] / change[index]))
            if candidate < fraction:
                fraction, blocking = candidate, index
        x += fraction * direction
        if blocking is not None and fraction < 1 - 1e-12:
            trial = working + [blocking]
            if np.linalg.matrix_rank(a[trial], tol=1e-10) != len(trial):
                raise RuntimeError("Dependent QP blocking constraint; original state retained")
            working = trial
    raise RuntimeError("Active-set QP iteration budget exhausted; no retry")
