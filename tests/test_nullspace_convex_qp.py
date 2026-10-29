"""Regression for the captured native failure and independent original-QP KKT checks."""

from pathlib import Path

import numpy as np
import pytest

from s2a.nullspace_convex_qp import solve_qp
from s2a.strict_convex_qp import solve_qp as original


def test_captured_failure_and_original_kkt():
    with np.load(Path(__file__).parent / "fixtures/preview_dependent_qp.npz", allow_pickle=False) as archive:
        h, g, a, b, x = [archive[k] for k in ("hessian", "linear", "constraints", "bounds", "initial")]
    with pytest.raises(RuntimeError, match="Dependent QP blocking"):
        original(h, g, a, b, x)
    result = solve_qp(h, g, a, b, x)
    active, dual = result["active"], result["multipliers"]
    slack = b - a @ result["solution"]
    assert slack.min() >= -1e-9
    assert dual.min() >= 0
    assert np.max(np.abs(h @ result["solution"] + g + a[active].T @ dual)) < 1e-8
    assert np.max(np.abs(slack[active] * dual)) < 1e-8
    assert result["solution"] @ h @ result["solution"] / 2 + g @ result["solution"] <= x @ h @ x / 2 + g @ x


def test_duplicate_faces_have_known_solution():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    result = solve_qp(np.eye(2), [-2, 0.3], a, np.ones(5), [0, 0])
    np.testing.assert_allclose(result["solution"], [1, -0.3], atol=1e-10, rtol=0)


def test_leaves_wrong_initial_active_face():
    result = solve_qp(np.eye(2), [0.2, 0.4], np.vstack((np.eye(2), -np.eye(2))), np.ones(4), [1, 1])
    np.testing.assert_allclose(result["solution"], [-0.2, -0.4], atol=1e-10, rtol=0)


def test_infeasible_initial_rejected():
    with pytest.raises(ValueError, match="Feasible"):
        solve_qp(np.eye(1), [0], [[1]], [1], [2])


def test_indefinite_objective_rejected():
    with pytest.raises(np.linalg.LinAlgError):
        solve_qp([[-1]], [0], [[1]], [1], [0])
