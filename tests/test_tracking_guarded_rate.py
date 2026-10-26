import numpy as np
import pytest

from s2a.tracking_guarded_rate import guarded_rate_command


def test_exact_root_and_infeasible_fallback():
    result = guarded_rate_command(np.eye(3), np.zeros(3), [0.2, 0, 0], np.zeros(3), [0.25, 0, 0], error_bound=0.01)
    assert result["tracking_guard_feasible"]
    assert result["change"] == pytest.approx(0.19)
    np.testing.assert_allclose(result["command"], [0.19, 0, 0])
    impossible = guarded_rate_command(np.eye(3), np.zeros(3), [1, 0, 0], np.zeros(3), [0.25, 0, 0])
    assert not impossible["tracking_guard_feasible"]
    np.testing.assert_array_equal(impossible["command"], [0.25, 0, 0])


def test_constant_map_and_no_additional_change_needed():
    result = guarded_rate_command(np.zeros((3, 3)), [0.001, 0, 0], np.zeros(3), [0.9, -0.9, 0], [1, -1, 0.2])
    assert result["change"] == 0.125 and result["tracking_guard_feasible"]
    np.testing.assert_allclose(result["command"], [1, -1, 0.125])
    result = guarded_rate_command(np.zeros((3, 3)), [0.01, 0, 0], np.zeros(3), np.zeros(3), np.ones(3) * 0.2)
    assert not result["tracking_guard_feasible"]


def test_piecewise_cross_coupling_matches_independent_polynomial_roots():
    rng = np.random.default_rng(42)
    for _ in range(60):
        matrix = rng.normal(size=(3, 3)) * 0.05
        previous = rng.uniform(-0.5, 0.5, 3)
        delta = rng.uniform(-0.25, 0.25, 3)
        command = previous + delta
        target = matrix @ command + rng.normal(size=3) * 0.002
        result = guarded_rate_command(matrix, np.zeros(3), target, previous, command)
        boundaries = sorted(set([0.125, 0.25, *[float(v) for v in np.abs(delta) if 0.125 < v < 0.25]]))
        feasible = []
        for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
            middle = (lo + hi) / 2
            slope = np.sign(delta) * (np.abs(delta) > middle)
            intercept = previous + np.where(np.abs(delta) > middle, 0, delta)
            a = matrix @ slope
            b = matrix @ intercept - target
            roots = np.polynomial.polynomial.polyroots([b @ b - 0.005**2, 2 * a @ b, a @ a]) if np.any(a) else []
            probes = [lo, hi, *[float(r.real) for r in roots if abs(r.imag) < 1e-10 and lo <= r.real <= hi]]
            feasible.extend(
                r
                for r in probes
                if np.linalg.norm(matrix @ (previous + np.clip(delta, -r, r)) - target) <= 0.005 + 1e-12
            )
        assert result["tracking_guard_feasible"] == bool(feasible)
        if feasible:
            assert result["change"] == pytest.approx(min(feasible), abs=1e-10)
            assert result["predicted_error"] <= 0.005 + 1e-12
        assert np.max(np.abs(result["command"] - previous)) <= 0.25 + 1e-12


def test_original_infeasible_command_rejected():
    with pytest.raises(ValueError, match="Original command"):
        guarded_rate_command(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3))


def test_single_tangent_feasible_point_is_retained():
    matrix = np.diag([1.0, 0, 0])
    result = guarded_rate_command(matrix, [0, 0.005, 0], [0.18, 0, 0], np.zeros(3), [0.25, 0, 0])
    assert result["tracking_guard_feasible"]
    assert result["change"] == pytest.approx(0.18, abs=1e-10)
    assert result["predicted_error"] <= 0.005 + 1e-12
