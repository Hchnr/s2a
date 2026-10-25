import numpy as np
from scipy.optimize import linprog
from scipy.sparse.linalg import aslinearoperator

from s2a.terminal_objective_solver import fit_opening_lad, lad_certificate, ridge_quadratic


def test_matrix_free_quadratic_matches_dense_solution():
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(35, 7)), rng.normal(size=35)
    d, evidence = ridge_quadratic(aslinearoperator(a), b, 0.1)
    np.testing.assert_allclose(d, np.linalg.solve(a.T @ a + 0.1 * np.eye(7), -a.T @ b), atol=1e-10)
    assert evidence["cg_info"] == 0
    assert evidence["relative_normal_residual"] < 1e-9


def test_lad_fit_and_dual_certificate_against_known_optimum():
    # Constant median at 0.2 remains optimal with the small ridge penalty:
    # the mass at 0.2 admits a subgradient balancing the ridge derivative.
    a = np.ones((5, 1))
    b = -np.array([0.1, 0.2, 0.2, 0.2, 0.9])
    weights = np.ones(5) / 5
    d, history = fit_opening_lad(aslinearoperator(a), b, weights, 0.01)
    optimum = float(weights @ abs(a[:, 0] * 0.2 + b) + 0.01 * 0.2**2)
    cert = history[-1]
    assert abs(d[0] - 0.2) < 1e-4
    assert cert["feasible"] and cert["dual"] <= optimum <= cert["primal"] + 1e-14
    assert cert["gap"] < 2e-4


def test_dual_bound_is_valid_away_from_optimum():
    a = np.array([[1.0], [2.0], [-1.0]])
    b, weights = np.array([0.3, -0.1, 0.8]), np.ones(3) / 3
    cert = lad_certificate(aslinearoperator(a), b, weights, np.array([3.0]), 0.1, 1e-4)
    # Independent LP lower bound for L1 without ridge is also a lower bound
    # for the regularized optimum; the feasible zero solution is an upper bound.
    lp = linprog(
        np.r_[0.0, weights],
        A_ub=np.vstack((np.c_[a, -np.eye(3)], np.c_[-a, -np.eye(3)])),
        b_ub=np.r_[-b, b],
        bounds=[(None, None)] + [(0, None)] * 3,
        method="highs",
    )
    assert lp.success
    assert cert["dual"] <= weights @ abs(b)
    assert cert["primal"] >= lp.fun
