"""Same-output success path and certified resolution of a real dependent blocker."""

from pathlib import Path

import numpy as np
import pytest

from s2a import dependency_safe_qp
from s2a.strict_convex_qp import solve_qp as original_solve


def test_successful_original_solution_is_unchanged():
    values = (np.eye(2), np.array([-0.2, 0.3]), np.vstack((np.eye(2), -np.eye(2))), np.ones(4), np.zeros(2))
    expected = original_solve(*values)
    actual = dependency_safe_qp.solve_qp(*values)
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_real_dependent_fixture_keeps_original_constraints_and_kkt():
    with np.load(Path(__file__).parent / "fixtures/preview_dependent_qp.npz", allow_pickle=False) as data:
        values = [data[k].copy() for k in ["hessian", "linear", "constraints", "bounds", "initial"]]
    saved = [v.copy() for v in values]
    with pytest.raises(RuntimeError, match="Dependent QP blocking"):
        original_solve(*values)
    result = dependency_safe_qp.solve_qp(*values)
    h, g, a, b, _ = values
    x = result["solution"]
    dual = np.zeros(len(a))
    dual[result["active"]] = result["multipliers"]
    assert np.min(b - a @ x) >= -1e-9 and dual.min() >= 0
    assert np.max(abs(h @ x + g + a.T @ dual)) <= 1e-8
    assert np.max(abs(dual * (b - a @ x))) <= 1e-8
    assert result["solver_route"] == "nullspace_after_dependent_blocker"
    for old, current in zip(saved, values, strict=True):
        np.testing.assert_array_equal(old, current)


def test_other_numerical_failure_is_not_silently_retried(monkeypatch):
    def failed(*args):
        raise RuntimeError("iteration budget exhausted")

    monkeypatch.setattr(dependency_safe_qp, "strict_solve", failed)
    monkeypatch.setattr(dependency_safe_qp, "nullspace_solve", lambda *args: pytest.fail("unregistered fallback"))
    with pytest.raises(RuntimeError, match="iteration budget exhausted"):
        dependency_safe_qp.solve_qp(None, None, None, None, None)
