import json

import numpy as np
import pytest
from current_coupled_independent import orientation_error, response, verify_inverse
from scipy.spatial.transform import Rotation
from test_rate_preview_active_set import contract

from s2a.current_coupled_response import build_response, nominal_prediction, rotation_spring, solve_translation


def plant(seed=42):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(7, 7))
    mass = a.T @ a + np.eye(7)
    jacobian = rng.normal(size=(6, 7))
    return mass, jacobian


def test_world_rotation_spring_jacobian_matches_column_cross_products():
    rng = np.random.default_rng(42)
    for _ in range(20):
        current = Rotation.random(random_state=rng).as_matrix()
        relative = Rotation.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix()
        goal = relative @ current
        forcing, stiffness = rotation_spring(relative)
        np.testing.assert_allclose(forcing, orientation_error(goal, current), atol=1e-14)
        for axis in range(3):
            delta = np.eye(3)[axis] * 1e-6
            derivative = (
                orientation_error(goal, Rotation.from_rotvec(delta).as_matrix() @ current)
                - orientation_error(goal, Rotation.from_rotvec(-delta).as_matrix() @ current)
            ) / 2e-6
            np.testing.assert_allclose(derivative, -stiffness[:, axis], atol=2e-10, rtol=0)


def test_coupled_matrix_exponential_matches_independent_finite_difference_ode():
    c = contract()
    mass, j = plant()
    relative = Rotation.from_rotvec([0.2, -0.15, 0.1]).as_matrix()
    produced = build_response(mass, j, c["kp"], c["kd"], relative)
    independent = response(mass, j, c["kp"], c["kd"], relative)
    for key in independent:
        np.testing.assert_allclose(produced[key], independent[key], atol=2e-9, rtol=0)


def test_known_coupled_endpoint_recovers_command_with_completed_innovation():
    c = contract()
    mass, j = plant()
    model = build_response(mass, j, c["kp"], c["kd"], Rotation.from_rotvec([0.1, 0.2, -0.1]).as_matrix())
    position, twist = np.array([0.4, -0.1, 0.9]), np.array([0.03, -0.01, 0.02, 0.1, -0.03, 0.05])
    wanted = np.array([0.15, -0.07, 0.1])
    expected = nominal_prediction(model, twist, wanted * 0.05)
    innovation = np.array([[0.0002, -0.0003, 0.0001], [0.003, 0.001, -0.002]])
    target = position + expected[:3] + innovation[0]
    target_velocity = expected[6:9] + innovation[1]
    result = solve_translation(position, twist, target, target_velocity, np.zeros(3), c, model, innovation=innovation)
    np.testing.assert_allclose(result["command"], wanted, atol=1e-10, rtol=0)
    independent = response(mass, j, c["kp"], c["kd"], model["relative_rotation_goal"])
    saved = json.loads(json.dumps(result, default=lambda value: value.tolist()))
    verify_inverse(saved, position, twist, target, target_velocity, np.zeros(3), c, independent, innovation)


def test_uncoupled_translation_reduces_to_original_constrained_single_period_ls():
    from pose_velocity_arm_independent_replay import PoseVelocityArmReplay

    from s2a.osc import OSCInterface

    c = contract()
    mass, j = np.eye(7), np.c_[np.eye(6), np.zeros(6)]
    model = build_response(mass, j, c["kp"], c["kd"], Rotation.from_rotvec([0.2, -0.1, 0.05]).as_matrix())
    v = np.array([0.03, -0.04, 0.02, 0.1, -0.2, 0.05])
    target, ref_velocity = np.array([0.001, -0.002, 0.003]), np.array([0.04, -0.01, 0.02])
    result = solve_translation(np.zeros(3), v, target, ref_velocity, np.zeros(3), c, model, innovation=np.zeros((2, 3)))
    interface = OSCInterface(*(c[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    baseline = PoseVelocityArmReplay(interface, c["kp"], c["kd"], 0.05, max_command_rate=5)
    desired = np.eye(4)
    desired[:3, 3] = target
    expected = baseline.convert(np.eye(4), desired, v, desired_linear_velocity=ref_velocity)
    np.testing.assert_allclose(result["command"], expected["command"][:3], atol=1e-12, rtol=0)


def test_saturated_coupled_commands_have_independent_kkt_and_actual_rate_bounds():
    rng = np.random.default_rng(42)
    c = contract()
    mass, j = plant()
    model = build_response(mass, j, c["kp"], c["kd"], Rotation.from_rotvec([0.2, 0.1, -0.1]).as_matrix())
    independent = response(mass, j, c["kp"], c["kd"], model["relative_rotation_goal"])
    for _ in range(15):
        position = rng.normal(0, 0.2, 3)
        twist = rng.normal(0, 0.05, 6)
        target = position + rng.normal(0, 0.01, 3)
        target_velocity = rng.normal(0, 0.05, 3)
        previous = rng.uniform(-1, 1, 3)
        innovation = rng.normal(0, [0.001, 0.02], (3, 2)).T
        result = solve_translation(position, twist, target, target_velocity, previous, c, model, innovation=innovation)
        assert np.max(np.abs(result["command"] - previous)) <= 0.25 + 1e-12
        verify_inverse(result, position, twist, target, target_velocity, previous, c, independent, innovation)


def test_invalid_mass_or_singular_robot_jacobian_is_rejected():
    c = contract()
    with pytest.raises(np.linalg.LinAlgError):
        build_response(-np.eye(7), np.c_[np.eye(6), np.zeros(6)], c["kp"], c["kd"], np.eye(3))
    with pytest.raises(ValueError, match="rank"):
        build_response(np.eye(7), np.zeros((6, 7)), c["kp"], c["kd"], np.eye(3))
