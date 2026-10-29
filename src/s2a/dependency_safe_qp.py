"""Preserve strict-QP results; handle only its dependent-blocker numerical failure."""

import numpy as np

from s2a.nullspace_convex_qp import solve_qp as nullspace_solve
from s2a.strict_convex_qp import solve_qp as strict_solve


def solve_qp(hessian, linear, constraints, bounds, initial):
    try:
        return strict_solve(hessian, linear, constraints, bounds, initial)
    except RuntimeError as error:
        if str(error) != "Dependent QP blocking constraint; original state retained":
            raise
        original_error = str(error)
    result = nullspace_solve(hessian, linear, constraints, bounds, initial)
    h, g, a, b = (np.asarray(v, dtype=float) for v in (hessian, linear, constraints, bounds))
    x = np.asarray(result["solution"])
    dual = np.zeros(len(a))
    dual[np.asarray(result["active"], dtype=int)] = result["multipliers"]
    slack = b - a @ x
    np.linalg.cholesky(h)
    if (
        slack.min() < -1e-9
        or dual.min() < -1e-9
        or np.max(abs(h @ x + g + a.T @ dual)) > 1e-8
        or np.max(abs(dual * slack)) > 1e-8
    ):
        raise RuntimeError("Dependency-safe QP failed original primal/dual/KKT gates")
    return dict(result, solver_route="nullspace_after_dependent_blocker", original_solver_error=original_error)
