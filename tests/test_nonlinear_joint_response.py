"""Numerical OSC: native torque semantics, causal rollout and integration accuracy."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.nonlinear_joint_response import rollout_nonlinear_joint_response


def _cartesian_state(q, velocity, *, mass=None, jacobian=None, bias=None, damping=None):
    """A synthetic Cartesian robot; rotation tests below use only its z axis."""
    pose = np.eye(4)
    pose[:3, 3] = q[:3]
    pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
    return dict(
        mass=np.eye(7) if mass is None else mass,
        jacobian=np.c_[np.eye(6), np.zeros(6)] if jacobian is None else jacobian,
        pose=pose,
        bias=np.zeros(7) if bias is None else bias,
        damping=np.zeros(7) if damping is None else damping,
    )


def _rollout(q, velocity, delta, dynamics, **kwargs):
    parameters = dict(
        initial_joint=q.copy(),
        kp=np.ones(6),
        kd=np.ones(6),
        torque_lower=np.full(7, -1e6),
        torque_upper=np.full(7, 1e6),
        dt=0.002,
        substeps=25,
        update_matrices=True,
        null_kp=0.0,
    )
    parameters.update(kwargs)
    return rollout_nonlinear_joint_response(q, velocity, delta, dynamics=dynamics, **parameters)


def test_single_step_world_sine_error_and_inertially_coupled_null_torque():
    # J=[I,0] leaves joint 7 task-redundant, but M couples its null torque
    # into joint 1. The 2x2 block gives a hand-computable Schur complement.
    mass = np.diag([2.0, 1.2, 1.3, 1.4, 1.5, 1.6, 3.0])
    mass[0, 6] = mass[6, 0] = 0.6
    q = np.array([0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 0.4])
    velocity = np.array([0.15, -0.04, 0.02, 0.0, 0.0, 0.07, -0.12])
    initial = q + np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0, -0.06])
    delta = np.array([0.02, -0.03, 0.01, 0.0, 0.0, 0.4])
    kp = np.array([100.0, 110.0, 120.0, 70.0, 80.0, 90.0])
    kd = np.array([20.0, 21.0, 22.0, 16.0, 17.0, 18.0])
    orientation = Rotation.from_euler("xyz", [0.2, -0.3, 0.1]).as_matrix()

    def dynamics(position, speed):
        state = _cartesian_state(position, speed, mass=mass)
        state["pose"][:3, :3] = orientation
        return state

    actual = _rollout(q, velocity, delta, dynamics, initial_joint=initial, kp=kp, kd=kd, null_kp=4.0, substeps=1)
    expected = np.zeros(7)
    expected[:3] = [1.88, 1.2, 1.3] * (kp[:3] * delta[:3] - kd[:3] * velocity[:3])
    expected[5] = 1.6 * (kp[5] * np.sin(0.4) - kd[5] * velocity[5])
    null_accel_0 = 4 * 0.03 - 4 * velocity[0]
    null_accel_6 = 4 * -0.06 - 4 * velocity[6]
    expected[0] += 0.12 * null_accel_0 + 0.6 * null_accel_6
    expected[6] = 0.6 * null_accel_0 + 3 * null_accel_6
    np.testing.assert_allclose(actual["requested_torque"][0], expected, atol=2e-13, rtol=0)
    next_velocity = velocity + 0.002 * np.linalg.solve(mass, expected)
    np.testing.assert_allclose(actual["joint_velocity"][0], next_velocity, atol=1e-14, rtol=0)
    np.testing.assert_allclose(actual["joint_position"][0], q + 0.002 * next_velocity, atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        actual["goal"][:3, :3], Rotation.from_rotvec(delta[3:]).as_matrix() @ orientation, atol=1e-15, rtol=0
    )


def test_bias_is_compensated_before_actuator_clipping_then_removed_from_plant():
    bias = np.array([0.8, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0])

    def dynamics(q, velocity):
        return _cartesian_state(q, velocity, bias=bias)

    actual = _rollout(
        np.zeros(7),
        np.zeros(7),
        [0.1, -0.1, 0.0, 0.0, 0.0, 1e-12],
        dynamics,
        kp=np.full(6, 5.0),
        torque_lower=-np.ones(7),
        torque_upper=np.ones(7),
        dt=0.1,
        substeps=1,
    )
    np.testing.assert_allclose(actual["requested_torque"][0, :2], [1.3, -1.3], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(actual["applied_torque"][0, :2], [1.0, -1.0])
    np.testing.assert_array_equal(actual["clipped"][0, :2], [True, True])
    np.testing.assert_allclose(actual["joint_velocity"][0, :2], [0.02, -0.02], atol=1e-15, rtol=0)


def test_viscous_damping_is_implicit_and_other_passive_force_is_added_once():
    mass, damping = np.eye(7), np.zeros(7)
    mass[6, 6], damping[6] = 2.0, 4.0
    passive = np.zeros(7)
    passive[6] = 0.6

    def dynamics(q, velocity):
        return dict(_cartesian_state(q, velocity, mass=mass, damping=damping), passive_extra=passive)

    velocity = np.zeros(7)
    velocity[6] = 1.0
    actual = _rollout(np.zeros(7), velocity, [0.0, 0.0, 0.0, 0.0, 0.0, 1e-12], dynamics, dt=0.1, substeps=2)
    first = (2.0 + 0.06) / 2.4
    second = (2 * first + 0.06) / 2.4
    np.testing.assert_allclose(actual["joint_velocity"][:, 6], [first, second], atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual["joint_position"][:, 6], [0.1 * first, 0.1 * (first + second)], atol=1e-15)
    np.testing.assert_array_equal(actual["requested_torque"][:, 6], 0.0)


def test_constant_matrices_make_both_modes_identical_with_state_dependent_forces():
    def dynamics(q, velocity):
        return _cartesian_state(q, velocity, bias=0.3 * q, damping=0.1 + q * q)

    q, velocity = np.zeros(7), np.zeros(7)
    q[0], q[5], velocity[0], velocity[6] = 0.3, 0.1, -0.02, 0.2
    delta = [0.1, 0.0, 0.0, 0.0, 0.0, 0.2]
    frozen = _rollout(q, velocity, delta, dynamics, update_matrices=False)
    updated = _rollout(q, velocity, delta, dynamics, update_matrices=True)
    for key in (
        "joint_position",
        "joint_velocity",
        "requested_torque",
        "applied_torque",
        "goal",
        "endpoint_pose",
        "endpoint_twist",
    ):
        np.testing.assert_array_equal(frozen[key], updated[key])


@pytest.mark.parametrize("update", [False, True])
def test_only_mass_and_jacobian_are_frozen_while_geometry_and_passive_force_advance(update):
    # One nonlinear Cartesian coordinate: x(q)=q+q^2, J=1+2q.
    # Its scalar OSC acceleration is e/J-v; passive acceleration is F/M.
    def dynamics(q, velocity):
        state = _cartesian_state(q, velocity)
        state["mass"][0, 0] = 1 + 4 * q[0]
        state["jacobian"][0, 0] = 1 + 2 * q[0]
        state["pose"][0, 3] = q[0] + q[0] ** 2
        state["passive_extra"] = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return state

    velocity = np.zeros(7)
    velocity[0] = 0.2
    actual = _rollout(
        np.zeros(7),
        velocity,
        [0.1, 0.0, 0.0, 0.0, 0.0, 1e-12],
        dynamics,
        dt=0.1,
        substeps=2,
        update_matrices=update,
    )
    first_q, first_v = 0.026, 0.26
    error = 0.1 - first_q - first_q**2
    acceleration = error / (1 + 2 * first_q if update else 1) - first_v
    acceleration += 0.7 / (1 + 4 * first_q if update else 1)
    second_v = first_v + 0.1 * acceleration
    np.testing.assert_allclose(actual["joint_velocity"][:, 0], [first_v, second_v], atol=1e-14, rtol=0)
    np.testing.assert_allclose(actual["joint_position"][:, 0], [first_q, first_q + 0.1 * second_v], atol=1e-14, rtol=0)
    np.testing.assert_allclose(actual["mass_change"], [0.0, 4 * first_q], atol=1e-15, rtol=0)


def test_semimplicit_translation_converges_to_existing_exact_frozen_response():
    q = np.array([0.2, -0.3, 0.4, 0.0, 0.0, 0.0, 0.1])
    velocity = np.array([-0.02, 0.01, 0.03, 0.0, 0.0, 0.0, 0.0])
    delta = np.array([0.02, -0.03, 0.01, 0.0, 0.0, 1e-12])
    kp, kd = np.full(6, 80.0), np.full(6, 16.0)
    linear = build_joint_response(np.eye(7), np.c_[np.eye(6), np.zeros(6)], kp, kd, q, q, null_kp=0.0, dt=0.002)
    exact = predict_joint_response(linear, velocity, delta, q - 100, q + 100, substeps=25, enforce_limits=False)
    expected = np.r_[exact["joint_position"][:3], exact["joint_velocity"][:3]]
    errors = []
    for dt, substeps in [(0.002, 25), (0.001, 50), (0.0005, 100)]:
        actual = _rollout(q, velocity, delta, _cartesian_state, kp=kp, kd=kd, dt=dt, substeps=substeps)
        observed = np.r_[actual["joint_position"][-1, :3], actual["joint_velocity"][-1, :3]]
        errors.append(np.linalg.norm(observed - expected))
    assert errors[-1] < 5e-4
    np.testing.assert_array_less(1.8, np.asarray(errors[:-1]) / errors[1:])
    np.testing.assert_array_less(np.asarray(errors[:-1]) / errors[1:], 2.2)


def test_goal_remains_at_period_start_anchor_during_inner_motion():
    q = np.zeros(7)
    q[0] = 0.4
    actual = _rollout(
        q,
        np.zeros(7),
        [0.1, 0.0, 0.0, 0.0, 0.0, 1e-12],
        _cartesian_state,
        kp=np.full(6, 2.0),
        kd=np.full(6, 0.3),
        dt=0.1,
        substeps=2,
    )
    assert actual["goal"][0, 3] == pytest.approx(0.5)
    np.testing.assert_allclose(actual["requested_torque"][:, 0], [0.2, 0.19], atol=1e-15, rtol=0)
    np.testing.assert_allclose(actual["joint_position"][:, 0], [0.402, 0.4059], atol=1e-15, rtol=0)


@pytest.mark.parametrize("update", [False, True])
def test_callback_sees_only_own_predicted_states_and_cannot_mutate_input_state(update):
    seen = []

    def dynamics(q, velocity):
        seen.append((q.copy(), velocity.copy()))
        state = _cartesian_state(q, velocity, mass=np.diag(1.0 + q * q), bias=0.2 * q)
        # A backend is given owned copies, so incidental input-buffer reuse must
        # not corrupt either the integrator or the original measured state.
        q[:] = 1000.0
        velocity[:] = -1000.0
        return state

    q, velocity = np.linspace(0.0, 0.3, 7), np.linspace(-0.1, 0.1, 7)
    delta = np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.1])
    initial, kp, kd = q + 0.02, np.full(6, 20.0), np.full(6, 4.0)
    originals = [value.copy() for value in (q, velocity, delta, initial, kp, kd)]
    actual = _rollout(
        q, velocity, delta, dynamics, initial_joint=initial, kp=kp, kd=kd, substeps=4, update_matrices=update
    )
    assert actual["dynamics_queries"] == len(seen) == 5
    np.testing.assert_array_equal(np.asarray([item[0] for item in seen]), np.vstack((q, actual["joint_position"])))
    np.testing.assert_array_equal(
        np.asarray([item[1] for item in seen]), np.vstack((velocity, actual["joint_velocity"]))
    )
    for original, after in zip(originals, (q, velocity, delta, initial, kp, kd), strict=True):
        np.testing.assert_array_equal(after, original)


def test_zero_rotation_latch_is_rejected_before_any_dynamics_query():
    def forbidden_dynamics(q, velocity):
        raise AssertionError("Latch state is missing, so dynamics must not be queried")

    with pytest.raises(ValueError, match="nonzero rotation"):
        _rollout(np.zeros(7), np.zeros(7), np.zeros(6), forbidden_dynamics)
