"""Analytic numerical checks; these do not evaluate robot or H12 responses."""

import numpy as np
import pytest

from s2a import accepted_point_sqp_core
from s2a.accepted_point_sqp_core import damped_bfgs, solve_epigraph_qp


def assert_original_units_kkt(result, B, g, A, b, expected):
    """Check the returned dual in the problem's original constraint units."""
    point = np.asarray(result["point"])
    dual = np.asarray(result["multipliers"])
    slack = A @ point + b
    gradient = np.r_[B @ point[:-1] + g, 1.0]
    assert result["qualified"], result
    np.testing.assert_allclose(point, expected, rtol=0, atol=2e-6)
    assert np.all(np.isfinite(dual))
    assert dual.shape == b.shape
    assert np.min(dual) >= 0
    assert np.min(slack) >= -2e-6
    np.testing.assert_allclose(A.T @ dual, gradient, rtol=0, atol=2e-6)
    np.testing.assert_allclose(dual * slack, 0, rtol=0, atol=2e-6)
    np.testing.assert_allclose(result["scaled_multipliers"], dual * result["row_scale"], rtol=1e-12)
    assert result["stationarity_relative"] <= 1e-6
    assert result["complementarity_relative"] <= 1e-6
    assert result["primal_min"] >= -1e-6
    return point, dual


def test_two_active_epigraph_faces_have_analytic_duals():
    # min t - d/2 + d^2, t >= max(1-d, 1+d), -2 <= d <= 2.
    # The kink is optimal; stationarity gives duals (1/4, 3/4).
    B = np.array([[2.0]])
    g = np.array([-0.5])
    A = np.array([[1, 1], [-1, 1], [1, 0], [-1, 0], [0, 1]], dtype=float)
    b = np.array([-1, -1, 2, 2, 0], dtype=float)
    original = [value.copy() for value in (B, g, A, b)]
    result = solve_epigraph_qp(B, g, A, b, 2.0)
    _, dual = assert_original_units_kkt(result, B, g, A, b, [0, 1])
    np.testing.assert_allclose(dual, [0.25, 0.75, 0, 0, 0], atol=2e-6)
    assert set(result["active_indices"]) == {0, 1}
    for value, before in zip((B, g, A, b), original, strict=True):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("scaling", [np.ones(4), np.array([0.001, 1000, 7, 0.03])])
def test_active_command_bound_and_row_scaling(scaling):
    # min t + d^2, t >= 2-d, -1 <= d <= .3.
    # The unconstrained minimizer .5 is cut off by the upper command bound.
    B = np.array([[2.0]])
    g = np.zeros(1)
    A = np.array([[1, 1], [1, 0], [-1, 0], [0, 1]], dtype=float)
    b = np.array([-2, 1, 0.3, 0], dtype=float)
    A = scaling[:, None] * A
    b = scaling * b
    result = solve_epigraph_qp(B, g, A, b, 3.0)
    _, dual = assert_original_units_kkt(result, B, g, A, b, [0.3, 1.7])
    np.testing.assert_allclose(dual * scaling, [1, 0, 0.4, 0], atol=2e-6)
    assert set(result["active_indices"]) == {0, 2}


def test_coupled_hessian_gives_analytic_interior_command():
    # On t=2-d1-d2, B d = (1,1)-g. This solution is interior to the box.
    B = np.array([[4.0, 1.0], [1.0, 2.0]])
    g = np.array([-0.2, 0.3])
    A = np.array(
        [[1, 1, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
        dtype=float,
    )
    b = np.array([-2, 1, 1, 1, 1, 0], dtype=float)
    result = solve_epigraph_qp(B, g, A, b, 3.0)
    _, dual = assert_original_units_kkt(result, B, g, A, b, [17 / 70, 8 / 35, 107 / 70])
    np.testing.assert_allclose(dual, [1, 0, 0, 0, 0, 0], atol=2e-6)


def test_nonunique_dual_with_redundant_epigraph_rows_still_certifies():
    # Two copies of t >= 1-d make the dual nonunique, while z stays unique.
    # A harmless constant row additionally checks a zero constraint gradient.
    B = np.array([[2.0]])
    g = np.array([-0.5])
    A = np.array([[1, 1], [3, 3], [-1, 1], [0, 1], [0, 0]], dtype=float)
    b = np.array([-1, -3, -1, 0, 1], dtype=float)
    result = solve_epigraph_qp(B, g, A, b, 2.0)
    _, dual = assert_original_units_kkt(result, B, g, A, b, [0, 1])
    assert dual[0] + 3 * dual[1] == pytest.approx(0.25, abs=2e-6)
    assert dual[2] == pytest.approx(0.75, abs=2e-6)
    assert set(result["active_indices"]) == {0, 1, 2}


def test_zero_epigraph_floor_can_be_the_only_active_constraint():
    B = np.array([[2.0]])
    g = np.array([0.5])
    A = np.array([[1, 0], [-1, 0], [0, 1]], dtype=float)
    b = np.array([1, 1, 0], dtype=float)
    result = solve_epigraph_qp(B, g, A, b, 1.0)
    _, dual = assert_original_units_kkt(result, B, g, A, b, [-0.25, 0])
    np.testing.assert_allclose(dual, [0, 0, 1], atol=2e-6)


def test_inconsistent_command_bounds_do_not_receive_a_certificate():
    # d >= 1 and d <= 0 are jointly impossible, independent of t.
    result = solve_epigraph_qp(
        np.eye(1),
        np.zeros(1),
        np.array([[1, 0], [-1, 0], [0, 1]], dtype=float),
        np.array([-1, 0, 0], dtype=float),
        1.0,
    )
    assert not result["qualified"]
    assert result["primal_min"] < -0.1
    assert result["qualification_failures"]


def test_iteration_limit_does_not_certify_a_nonstationary_start():
    result = solve_epigraph_qp(
        np.array([[2.0]]),
        np.zeros(1),
        np.array([[1, 1], [1, 0], [-1, 0], [0, 1]], dtype=float),
        np.array([-2, 1, 0.3, 0], dtype=float),
        3.0,
        max_iterations=1,
    )
    assert not result["qualified"]
    assert not result["solver_success"]
    assert result["solver_nit"] <= 1
    assert result["qualification_failures"]


def test_certificate_failure_cannot_promote_a_successful_qp_solver(monkeypatch):
    calls = 0

    def fail_certificate(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected certificate failure")

    monkeypatch.setattr(accepted_point_sqp_core, "nnls", fail_certificate)
    # SLSQP still solves a real analytic QP; only its certificate fails.
    result = solve_epigraph_qp(
        np.array([[2.0]]),
        np.array([0.5]),
        np.array([[1, 0], [-1, 0], [0, 1]], dtype=float),
        np.array([1, 1, 0], dtype=float),
        1.0,
    )
    assert result["solver_success"]
    np.testing.assert_allclose(result["point"], [-0.25, 0], atol=2e-6)
    assert calls == 1
    assert not result["qualified"] and not result["nnls_success"]
    assert "injected certificate failure" in result["nnls_error"]
    assert "nnls_failed" in result["qualification_failures"]
    assert result["multipliers"] is None


@pytest.mark.parametrize("B", [np.array([[-1.0]]), np.array([[np.nan]])])
def test_qp_rejects_invalid_curvature(B):
    with pytest.raises(ValueError):
        solve_epigraph_qp(B, np.zeros(1), np.array([[0.0, 1.0]]), np.zeros(1), 1.0)


def test_qp_rejects_constraint_shape_mismatch():
    with pytest.raises(ValueError):
        solve_epigraph_qp(np.eye(2), np.zeros(2), np.ones((1, 2)), np.zeros(1), 1.0)


def test_bfgs_satisfies_secant_equation_and_preserves_spd():
    B = np.array([[2.0, 0.2], [0.2, 1.5]])
    before = B.copy()
    H = np.array([[4.0, 1.0], [1.0, 3.0]])
    s = np.array([0.25, -0.5])
    y = H @ s
    result = damped_bfgs(B, s, y)
    updated = np.asarray(result["matrix"])
    assert result["updated"]
    assert not result["damping_applied"]
    assert result["theta"] == 1.0
    np.testing.assert_allclose(updated @ s, y, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(updated, updated.T)
    assert np.min(np.linalg.eigvalsh(updated)) > 0
    np.testing.assert_array_equal(B, before)


@pytest.mark.parametrize("step_scale", [1.0, 1e-8])
def test_bfgs_retains_curvature_between_two_accepted_steps(step_scale):
    # Orthogonal quadratic secants recover the two independent curvatures.
    # Valid small displacements must not introduce an implicit reset/threshold.
    first = damped_bfgs(np.eye(2), step_scale * np.array([1.0, 0.0]), step_scale * np.array([2.0, 0.0]))
    second = damped_bfgs(first["matrix"], step_scale * np.array([0.0, 1.0]), step_scale * np.array([0.0, 3.0]))
    assert first["updated"] and second["updated"]
    np.testing.assert_allclose(second["matrix"], np.diag([2.0, 3.0]), atol=1e-12)
    np.testing.assert_allclose(first["matrix"], np.diag([2.0, 1.0]), atol=1e-12)


def test_powell_damping_handles_negative_observed_curvature():
    B = np.array([[4.0, 1.0], [1.0, 2.0]])
    s = np.array([0.5, -0.25])
    Bs = B @ s
    # y = -B s gives theta=.4 and the damped secant r=.2 B s.
    result = damped_bfgs(B, s, -Bs)
    updated = np.asarray(result["matrix"])
    assert result["updated"] and result["damping_applied"]
    assert result["theta"] == pytest.approx(0.4)
    assert result["sTr"] == pytest.approx(0.175)
    np.testing.assert_allclose(updated @ s, 0.2 * Bs, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(updated)) > 0
    np.testing.assert_array_equal(updated, updated.T)


@pytest.mark.parametrize(
    ("s", "y", "minimum_step", "reason"),
    [
        ([0.0, 0.0], [1.0, 2.0], 0.0, "zero_step"),
        ([3e-12, 4e-12], [6e-12, 12e-12], 1e-11, "step_below_minimum"),
        ([np.nan, 1.0], [1.0, 2.0], 0.0, None),
        ([1.0, 0.0], [np.inf, 2.0], 0.0, None),
    ],
)
def test_bfgs_unusable_pairs_leave_old_curvature_untouched(s, y, minimum_step, reason):
    B = np.array([[2.0, 0.5], [0.5, 1.0]])
    before = B.copy()
    result = damped_bfgs(B, np.array(s), np.array(y), minimum_step=minimum_step)
    assert not result["updated"]
    assert result["reason"]
    if reason is not None:
        assert result["reason"] == reason
    np.testing.assert_array_equal(result["matrix"], before)
    np.testing.assert_array_equal(B, before)


def test_bfgs_rejects_non_symmetric_curvature_and_bad_pair_shape():
    with pytest.raises(ValueError):
        damped_bfgs(np.array([[2.0, 0.1], [0.2, 1.0]]), np.ones(2), np.ones(2))
    with pytest.raises(ValueError):
        damped_bfgs(np.eye(2), np.ones(3), np.ones(2))
