import numpy as np
import pytest

from s2a.bounded_response_inverse import translation_command


def solve(j, target, **changes):
    config = dict(
        dt=0.05,
        input_min=-np.ones(3),
        input_max=np.ones(3),
        output_min=-np.ones(3),
        output_max=np.ones(3),
        previous_command=np.zeros(3),
        max_rate=100,
    )
    config.update(changes)
    return translation_command(np.zeros(6), j, target, np.zeros(3), **config)


def test_feasible_coupled_inverse_recovers_known_action():
    j = np.vstack([np.array([[1, 0.3, 0], [0, 1, 0.2], [0.1, 0, 1]]), np.zeros((3, 3))])
    wanted = np.array([0.2, -0.3, 0.1])
    out = solve(j, j[:3] @ wanted)
    np.testing.assert_allclose(out["command"], wanted, atol=1e-14)
    assert out["squared_objective"] < 1e-26


def test_coupled_bound_optimum_improves_over_post_inverse_clipping():
    j = np.zeros((6, 3))
    j[:3] = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    out = solve(
        j,
        np.array([1, 0, 0]),
        input_min=np.array([-0.1, -1, -1]),
        input_max=np.array([0.1, 1, 1]),
        output_min=np.array([-0.1, -1, -1]),
        output_max=np.array([0.1, 1, 1]),
    )
    np.testing.assert_allclose(out["command"], [0.1, 0.45, 0], atol=1e-13)
    clipped_error = np.linalg.norm(j[:3] @ np.array([0.1, 0, 0]) - np.array([1, 0, 0])) ** 2
    assert out["squared_objective"] < clipped_error
    grid = np.linspace(-1, 1, 2001)
    independent = min((0.1 + y - 1) ** 2 + y * y for y in grid)
    assert abs(out["squared_objective"] - independent) < 1e-14


def test_rate_bound_and_physical_units_are_enforced_inside_solve():
    j = np.vstack([np.eye(3), np.zeros((3, 3))])
    out = solve(j, np.ones(3), output_min=-np.ones(3) * 0.05, output_max=np.ones(3) * 0.05, max_rate=5)
    np.testing.assert_allclose(out["command"], 0.25, atol=1e-14)
    np.testing.assert_allclose(out["physical_delta"], 0.0125, atol=1e-14)


def test_unreachable_previous_command_is_rejected():
    with pytest.raises(ValueError, match="feasible previous"):
        solve(np.zeros((6, 3)), np.zeros(3), previous_command=np.array([2, 0, 0]))


def test_position_and_scaled_velocity_joint_objective_can_conflict():
    j = np.vstack([np.eye(3), np.eye(3)])
    out = translation_command(
        np.zeros(6),
        j,
        np.array([1, 0, 0]),
        np.array([-20, 0, 0]),
        dt=0.05,
        input_min=-np.ones(3),
        input_max=np.ones(3),
        output_min=-np.ones(3),
        output_max=np.ones(3),
        previous_command=np.zeros(3),
        max_rate=100,
    )
    np.testing.assert_allclose(out["command"], np.zeros(3), atol=1e-14)
    assert out["squared_objective"] == pytest.approx(2)


def test_rank_deficient_response_stays_finite_and_feasible():
    out = solve(np.zeros((6, 3)), np.ones(3), max_rate=5)
    assert np.isfinite(out["command"]).all()
    assert np.all(np.abs(out["command"]) <= 0.25)
    assert out["squared_objective"] == pytest.approx(3)
