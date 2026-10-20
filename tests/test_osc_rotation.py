import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from s2a.osc import OSCInterface
from s2a.osc_rotation import bounded_osc_rotation_inverse, predict_osc_orientation


def interface():
    return OSCInterface(-np.ones(6), np.ones(6), -np.array([0.05] * 3 + [0.5] * 3), np.array([0.05] * 3 + [0.5] * 3))


def independent_endpoint(measured, delta, omega, kp, kd, dt):
    """Adaptive quaternion ODE, independent of the implementation's Lie midpoint."""
    goal = Rotation.from_rotvec(delta).as_matrix() @ measured

    def system(_, state):
        q, w = state[:4], state[4:]
        current = Rotation.from_quat(q).as_matrix()
        spring = 0.5 * sum(np.cross(current[:, i], goal[:, i]) for i in range(3))
        qdot = np.r_[0.5 * (q[3] * w + np.cross(w, q[:3])), -0.5 * np.dot(w, q[:3])]
        return np.r_[qdot, kp * spring - kd * w]

    solution = solve_ivp(
        system,
        (0, dt),
        np.r_[Rotation.from_matrix(measured).as_quat(), omega],
        rtol=1e-11,
        atol=1e-12,
    )
    assert solution.success
    return Rotation.from_quat(solution.y[:4, -1]).as_matrix(), solution.y[4:, -1]


def error(desired, actual):
    return np.linalg.norm(Rotation.from_matrix(desired @ actual.T).as_rotvec())


def test_prediction_matches_independent_sin_spring_dynamics_with_world_velocity():
    measured = Rotation.from_rotvec([0.7, -0.3, 0.4]).as_matrix()
    delta, omega = np.array([0.4, -0.35, 0.3]), np.array([0.4, -0.2, 0.3])
    kp, kd, dt = np.array([120, 150, 180]), np.array([22, 25, 27]), 0.05
    actual, actual_velocity = independent_endpoint(measured, delta, omega, kp, kd, dt)
    predicted, predicted_velocity = predict_osc_orientation(measured, delta, omega, kp=kp, kd=kd, dt=dt)
    assert error(actual, predicted) < 2e-5
    np.testing.assert_allclose(predicted_velocity, actual_velocity, atol=3e-4, rtol=0)


def test_feasible_inverse_hits_independent_endpoint_and_keeps_world_frame():
    measured = Rotation.from_rotvec([0.6, -0.4, 0.2]).as_matrix()
    omega = np.array([0.2, -0.15, 0.3])
    kp, kd, dt = np.full(3, 150.0), np.full(3, 2 * np.sqrt(150)), 0.05
    desired, _ = independent_endpoint(measured, [0.32, -0.25, 0.20], omega, kp, kd, dt)
    result = bounded_osc_rotation_inverse(measured, desired, omega, interface=interface(), kp=kp, kd=kd, dt=dt)
    after, _ = independent_endpoint(measured, result["sent_delta"], omega, kp, kd, dt)
    assert result["solver_converged"]
    assert result["predicted_endpoint_error_rad"] < 1e-7
    assert error(desired, after) < 2e-5
    assert not result["interface_bound_active"]


def test_large_rotation_is_bounded_and_improves_independent_endpoint_over_baseline():
    measured = Rotation.from_rotvec([0.5, 0.3, -0.7]).as_matrix()
    axis = np.array([0.8, -0.4, 0.3])
    axis /= np.linalg.norm(axis)
    desired = Rotation.from_rotvec(0.35 * axis).as_matrix() @ measured
    kp, kd, dt = np.full(3, 150.0), np.full(3, 2 * np.sqrt(150)), 0.05
    result = bounded_osc_rotation_inverse(measured, desired, np.zeros(3), interface=interface(), kp=kp, kd=kd, dt=dt)
    baseline, _ = independent_endpoint(measured, 0.35 * axis, np.zeros(3), kp, kd, dt)
    actual, _ = independent_endpoint(measured, result["sent_delta"], np.zeros(3), kp, kd, dt)
    assert np.all(np.abs(result["command"]) <= 1)
    assert result["interface_bound_active"]
    assert result["predicted_endpoint_error_rad"] > 0.1  # Explicitly unreachable in one interval.
    assert error(desired, actual) < error(desired, baseline) - 0.015
    assert result["predicted_endpoint_error_rad"] <= result["continuous_initial_seed_error_rad"] + 1e-10


def test_rate_interval_is_enforced_inside_inverse_for_reversing_motion():
    previous = np.array([0.8, -0.6, 0.2])
    measured = np.eye(3)
    desired = Rotation.from_rotvec([-0.3, 0.2, -0.1]).as_matrix()
    result = bounded_osc_rotation_inverse(
        measured,
        desired,
        np.array([0.2, -0.1, 0.3]),
        interface=interface(),
        kp=np.full(3, 150),
        kd=np.full(3, 25),
        dt=0.05,
        previous_command=previous,
        max_command_rate=5,
    )
    assert np.all(np.abs(result["command"] - previous) <= 0.25 + 1e-12)
    assert result["rate_bound_active"]
    np.testing.assert_allclose(result["physical_lower_bound"], (previous - 0.25) * 0.5)


def test_zero_command_preserves_current_goal_and_continues_motion():
    measured = Rotation.from_rotvec([0.4, -0.2, 0.3]).as_matrix()
    goal = Rotation.from_rotvec([0, 0, 0.35]).as_matrix() @ measured
    kp, kd, dt = np.full(3, 150), np.full(3, 25), 0.05
    predicted, _ = predict_osc_orientation(
        measured, np.zeros(3), np.zeros(3), kp=kp, kd=kd, dt=dt, current_goal_rotation=goal
    )
    independent, _ = independent_endpoint(measured, [0, 0, 0.35], np.zeros(3), kp, kd, dt)
    assert error(predicted, measured) > 0.03
    assert error(predicted, independent) < 2e-5
    result = bounded_osc_rotation_inverse(
        measured,
        predicted,
        np.zeros(3),
        interface=interface(),
        kp=kp,
        kd=kd,
        dt=dt,
        current_goal_rotation=goal,
    )
    assert result["zero_command_preserved_goal"]
    assert not result["zero_goal_epsilon_used"]
    np.testing.assert_array_equal(result["command"], np.zeros(3))
    np.testing.assert_allclose(result["applied_goal_rotation"], goal, atol=1e-14)


def test_explicit_nonzero_reanchor_compares_against_old_goal_and_obeys_rate_bounds():
    goal = Rotation.from_rotvec([0.3, 0, 0]).as_matrix()
    result = bounded_osc_rotation_inverse(
        np.eye(3),
        np.eye(3),
        np.zeros(3),
        interface=interface(),
        kp=np.full(3, 150),
        kd=np.full(3, 25),
        dt=0.05,
        current_goal_rotation=goal,
        previous_command=np.zeros(3),
        max_command_rate=5,
        zero_goal_epsilon_rad=1e-10,
    )
    assert result["zero_goal_epsilon_used"]
    assert result["zero_command_candidate_available"]
    assert not result["zero_command_preserved_goal"]
    assert 0 < np.linalg.norm(result["sent_delta"]) <= 1.001e-10
    assert result["predicted_endpoint_error_rad"] < 1e-10
    assert np.all(np.abs(result["command"]) <= 0.25)


def test_unknown_previous_goal_is_not_silently_treated_as_current_orientation():
    with pytest.raises(ValueError, match="requires current_goal_rotation"):
        predict_osc_orientation(np.eye(3), np.zeros(3), np.zeros(3), kp=np.full(3, 150), kd=np.full(3, 25), dt=0.05)
    with pytest.raises(ValueError, match="requires current_goal_rotation"):
        bounded_osc_rotation_inverse(
            np.eye(3), np.eye(3), np.zeros(3), interface=interface(), kp=np.full(3, 150), kd=np.full(3, 25), dt=0.05
        )


def test_float32_inputs_and_offset_affine_neutral_preserve_goal_without_epsilon():
    offset = OSCInterface(
        [-1, -1, -1, -2, -1, 0],
        [1, 1, 1, 2, 3, 4],
        [-0.05, -0.05, -0.05, -0.25, -0.5, -0.75],
        [0.05, 0.05, 0.05, 0.75, 0.5, 0.25],
    )
    neutral = np.array([-1, 1, 3], dtype=np.float32)
    goal = Rotation.from_rotvec([0.3, 0, 0]).as_matrix()
    result = bounded_osc_rotation_inverse(
        np.eye(3, dtype=np.float32),
        np.eye(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        interface=offset,
        kp=np.full(3, 150, dtype=np.float32),
        kd=np.full(3, 25, dtype=np.float32),
        dt=0.05,
        current_goal_rotation=goal,
        previous_command=neutral,
        max_command_rate=0.1,
    )
    assert result["command_dtype"] == "float64"
    assert result["zero_command_preserved_goal"]
    assert not result["zero_goal_epsilon_used"]
    np.testing.assert_array_equal(result["command"], neutral)
    np.testing.assert_array_equal(result["sent_delta"], np.zeros(3))
    assert result["predicted_endpoint_error_rad"] > 0.03
    assert np.all(np.abs(result["command"] - neutral) <= 0.005 + 1e-12)


def test_isotropic_model_is_covariant_under_world_frame_rotation():
    measured = Rotation.from_rotvec([0.2, -0.3, 0.5]).as_matrix()
    world_change = Rotation.from_rotvec([-0.4, 0.6, 0.2]).as_matrix()
    delta, omega = np.array([0.2, -0.1, 0.3]), np.array([0.1, 0.2, -0.15])
    kwargs = dict(kp=np.full(3, 150), kd=np.full(3, 25), dt=0.05)
    original, velocity = predict_osc_orientation(measured, delta, omega, **kwargs)
    transformed, changed_velocity = predict_osc_orientation(
        world_change @ measured, world_change @ delta, world_change @ omega, **kwargs
    )
    np.testing.assert_allclose(transformed, world_change @ original, atol=1e-12)
    np.testing.assert_allclose(changed_velocity, world_change @ velocity, atol=1e-12)


@pytest.mark.parametrize("field,value", [("dt", 0), ("integration_steps", 0), ("max_nfev", 0)])
def test_invalid_solver_contract_is_rejected(field, value):
    options = dict(interface=interface(), kp=np.full(3, 150), kd=np.full(3, 25), dt=0.05)
    options[field] = value
    with pytest.raises(ValueError):
        bounded_osc_rotation_inverse(np.eye(3), np.eye(3), np.zeros(3), **options)


def test_rejects_missing_rate_history_and_nonmonotone_goal_domain():
    options = dict(interface=interface(), kp=np.full(3, 150), kd=np.full(3, 25), dt=0.05)
    with pytest.raises(ValueError, match="requires previous"):
        bounded_osc_rotation_inverse(np.eye(3), np.eye(3), np.zeros(3), max_command_rate=5, **options)
    options["interface"] = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    with pytest.raises(ValueError, match="monotone"):
        bounded_osc_rotation_inverse(np.eye(3), np.eye(3), np.zeros(3), **options)
