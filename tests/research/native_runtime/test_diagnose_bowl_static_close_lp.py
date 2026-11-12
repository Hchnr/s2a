"""Capture LP residuals and respect explicit feasibility tolerances."""

import numpy as np

from scripts.simulation.runtime.diagnose_bowl_static_close_lp import linear_problem, solve_record


def test_reconstructs_infinity_constraints_and_records_both_solver_modes():
    rows = np.zeros((1, 4))
    a, b = linear_problem((rows, np.ones(1)), (rows, np.ones(1)), [0.01, -0.02, 0.03])
    for tolerance in ("default", 1e-9):
        result = solve_record(a, b, 0.02, tolerance)
        assert result["success"] and result["status"] == 0
        assert result["maximum_inequality_violation_m"] < 1e-10
        assert result["maximum_bound_violation_m"] < 1e-10
        assert result["x"][4] < 1e-10
