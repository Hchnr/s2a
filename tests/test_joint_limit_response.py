"""Free OSC equivalence, inertial coupling and unilateral release requirements."""

import numpy as np
import pytest

from s2a.current_coupled_response import build_response
from s2a.joint_limit_response import build_joint_response, predict_joint_response, project_box


def test_free_joint_model_matches_independent_task_osc_transition():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = a.T @ a + np.eye(7)
    j = rng.normal(size=(6, 7))
    q, velocity, initial = rng.normal(size=(3, 7))
    delta = rng.normal(size=6) * 0.01
    kp, kd = np.array([150.0] * 3 + [80.0] * 3), np.array([24.0] * 3 + [18.0] * 3)
    model = build_joint_response(mass, j, kp, kd, q, initial, null_kp=10.0, dt=0.002)
    actual = predict_joint_response(model, velocity, delta, q - 100, q + 100, substeps=25, enforce_limits=False)
    task_model = build_response(mass, j, kp, kd, np.eye(3))
    expected = task_model["initial_velocity_response"] @ (j @ velocity) + task_model["goal_forcing_response"] @ delta
    np.testing.assert_allclose(
        np.r_[actual["task_displacement"], actual["task_velocity"]], expected, atol=1e-11, rtol=0
    )


def test_mass_projection_preserves_coupled_optimum_not_component_clipping():
    # Holding v0 at upper=1 shifts v1 by +0.5 to minimize the coupled kinetic metric.
    actual = project_box([[2.0, 1.0], [1.0, 2.0]], [2.0, 0.0], [-10.0, -10.0], [1.0, 10.0])
    np.testing.assert_allclose(actual, [1.0, 0.5], atol=1e-10, rtol=0)


def test_upper_limit_blocks_outward_motion_but_allows_inward_release():
    mass, j = np.eye(7), np.c_[np.eye(6), np.zeros(6)]
    q = np.zeros(7)
    model = build_joint_response(mass, j, np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002)
    lower, upper = np.full(7, -1.0), np.full(7, 1.0)
    upper[0] = 0
    outward = predict_joint_response(
        model, np.zeros(7), [0.1, 0, 0, 0, 0, 0], lower, upper, substeps=25, enforce_limits=True
    )
    inward = predict_joint_response(
        model, np.zeros(7), [-0.1, 0, 0, 0, 0, 0], lower, upper, substeps=25, enforce_limits=True
    )
    assert outward["joint_position"][0] <= 1e-12
    assert abs(outward["joint_velocity"][0]) < 1e-10
    assert outward["position_projected_steps"] > 0
    assert inward["joint_position"][0] < -0.001
    assert inward["joint_velocity"][0] < 0
    assert inward["position_projected_steps"] == 0


def test_penetration_is_reported_and_invalid_ranges_rejected():
    q = np.zeros(7)
    q[6] = 1.0004
    model = build_joint_response(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6), np.ones(6), q, q, null_kp=10.0, dt=0.002
    )
    result = predict_joint_response(
        model, np.zeros(7), np.zeros(6), -np.ones(7), np.ones(7), substeps=1, enforce_limits=True
    )
    assert result["initial_max_penetration_rad"] == pytest.approx(0.0004)
    assert result["joint_position"][6] <= 1.0
    with pytest.raises(ValueError, match="ranges"):
        predict_joint_response(
            model, np.zeros(7), np.zeros(6), np.ones(7), -np.ones(7), substeps=1, enforce_limits=True
        )
