import copy
import json

import numpy as np
import pytest
from current_coupled_independent import response as independent_response
from full_coupled_pose_independent import verify_pose_solution
from scipy.spatial.transform import Rotation
from test_current_coupled_response import plant
from test_rate_preview_active_set import contract

from s2a.current_coupled_state import rotation_goal_from_command
from s2a.full_coupled_pose_inverse import FullCoupledPoseModel, projected_gradient, solve_pose, task_objective
from s2a.osc import OSCInterface


def setup():
    c = contract()
    m, j = plant()
    interface = OSCInterface(*(c[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    return c, FullCoupledPoseModel(m, j, c["kp"], c["kd"]), interface


def independent_prediction(model, delta, twist, fixed=None):
    relative = Rotation.from_rotvec(delta[3:]).as_matrix() if fixed is None else fixed
    response = independent_response(model.base["mass"], model.base["jacobian"], model.kp, model.kd, relative)
    return (
        response["initial_velocity_response"] @ twist
        + response["goal_forcing_response"] @ np.r_[delta[:3], response["rotation_forcing"]]
    )


def endpoint(measured, prediction, bias):
    pose = measured.copy()
    pose[:3, 3] += prediction[:3] + bias[:3]
    pose[:3, :3] = (
        Rotation.from_rotvec(bias[3:6]) * Rotation.from_rotvec(prediction[3:6]) * Rotation.from_matrix(measured[:3, :3])
    ).as_matrix()
    return pose, prediction[6:] + bias[6:] / 0.05


def test_goal_dependent_frechet_prediction_and_derivatives_match_ode_and_perturbations():
    _, model, _ = setup()
    delta, twist = np.array([0.008, -0.004, 0.006, 0.2, -0.12, 0.08]), np.array([0.03, -0.01, 0.02, 0.1, -0.05, 0.07])
    prediction, derivative = model.response(delta, twist, derivatives=True)
    np.testing.assert_allclose(prediction, independent_prediction(model, delta, twist), atol=2e-9, rtol=0)
    for axis in range(6):
        increment = np.eye(6)[axis] * 1e-6
        difference = (model.response(delta + increment, twist) - model.response(delta - increment, twist)) / 2e-6
        np.testing.assert_allclose(derivative[:, axis], difference, atol=3e-8, rtol=0)


def test_noncommuting_world_rotation_bias_objective_gradient_matches_finite_difference():
    _, model, interface = setup()
    measured = np.eye(4)
    measured[:3, :3] = Rotation.from_rotvec([0.4, -0.3, 0.2]).as_matrix()
    desired = measured.copy()
    desired[:3, :3] = Rotation.from_rotvec([0.1, -0.05, 0.03]).as_matrix() @ measured[:3, :3]
    desired[:3, 3] = [0.002, -0.003, 0.001]
    bias = np.array([0.0002, -0.0001, 0.0003, 0.03, -0.01, 0.02, 0.0003, -0.0002, 0.0001, 0.001, -0.002, 0.003])
    command, twist, target_twist = (
        np.array([0.1, -0.15, 0.08, 0.2, -0.13, 0.1]),
        np.arange(6) * 0.01,
        np.arange(6) * -0.02,
    )

    def objective(value):
        return task_objective(model, value, interface, measured, desired, twist, target_twist, bias)

    analytic = objective(command)["gradient"]
    numeric = np.array(
        [
            (
                objective(command + np.eye(6)[axis] * 1e-6)["objective"]
                - objective(command - np.eye(6)[axis] * 1e-6)["objective"]
            )
            / 2e-6
            for axis in range(6)
        ]
    )
    np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=1e-7)


def test_known_full_pose_and_world_twist_recovers_six_commands_with_bias():
    c, model, interface = setup()
    measured = np.eye(4)
    measured[:3, :3] = Rotation.from_rotvec([0.3, 0.2, -0.1]).as_matrix()
    measured[:3, 3] = [0.4, -0.1, 0.9]
    wanted = np.array([0.15, -0.12, 0.08, 0.1, -0.08, 0.05])
    twist = np.array([0.03, -0.01, 0.02, 0.1, -0.03, 0.05])
    bias = np.array([0.0002, -0.0003, 0.0001, 0.03, -0.02, 0.01, 0.0001, -0.0002, 0.0003, 0.001, -0.002, 0.003])
    desired, target_twist = endpoint(measured, independent_prediction(model, interface.decode(wanted), twist), bias)
    result = solve_pose(
        model, measured, desired, twist, target_twist, np.zeros(6), measured[:3, :3], c, innovation=bias
    )
    assert result["goal_mode"] == "reset" and not result["global_optimality_claim"]
    np.testing.assert_allclose(result["command"], wanted, atol=2e-6, rtol=0)
    assert result["objective"] < 1e-12


def test_zero_latch_is_a_distinct_feasible_goal_not_identity():
    c, model, interface = setup()
    measured, twist, bias = np.eye(4), np.zeros(6), np.zeros(12)
    previous_goal = Rotation.from_rotvec([0.35, -0.15, 0.2]).as_matrix()
    wanted = np.array([0.12, -0.05, 0.09, 0, 0, 0])
    desired, target_twist = endpoint(
        measured, independent_prediction(model, interface.decode(wanted), twist, previous_goal), bias
    )
    result = solve_pose(model, measured, desired, twist, target_twist, np.zeros(6), previous_goal, c, innovation=bias)
    assert result["goal_mode"] == "latch"
    np.testing.assert_array_equal(result["physical_delta"][3:], np.zeros(3))
    np.testing.assert_allclose(result["command"], wanted, atol=2e-6, rtol=0)


def test_exact_zero_reset_has_explicit_nonzero_native_representative():
    c, model, interface = setup()
    previous_goal = Rotation.from_rotvec([0.3, -0.1, 0.2]).as_matrix()
    result = solve_pose(
        model, np.eye(4), np.eye(4), np.zeros(6), np.zeros(6), np.zeros(6), previous_goal, c, innovation=np.zeros(12)
    )
    assert result["goal_mode"] == "reset" and result["reset_representative_adjusted"]
    assert np.any(result["physical_delta"][3:] != 0)
    goal, _ = rotation_goal_from_command(
        result["command"], interface, np.eye(3), previous_goal, matrix_decoder=lambda _: np.eye(3)
    )
    np.testing.assert_array_equal(goal, np.eye(3))
    assert result["max_projected_gradient"] < 2e-6


def test_saturated_solution_has_actual_six_axis_bounds_and_independent_ode_kkt():
    c, model, interface = setup()
    measured, desired = np.eye(4), np.eye(4)
    desired[:3, 3] = [0.1, -0.12, 0.08]
    desired[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.1]).as_matrix()
    twist, target_twist, bias = np.arange(6) * 0.01, np.arange(6) * -0.02, np.zeros(12)
    previous = np.array([0.7, -0.6, 0.4, 0.3, -0.4, 0.2])
    result = solve_pose(model, measured, desired, twist, target_twist, previous, np.eye(3), c, innovation=bias)
    assert np.all(np.abs(result["command"] - previous) <= 0.25 + 1e-12)

    def independent_objective(command):
        prediction = independent_prediction(model, interface.decode(command), twist, result["fixed_relative_goal"])
        predicted_pose, predicted_twist = endpoint(measured, prediction, bias)
        residual = np.r_[
            (desired[:3, 3] - predicted_pose[:3, 3]) / 0.01,
            Rotation.from_matrix(desired[:3, :3] @ predicted_pose[:3, :3].T).as_rotvec() / 0.05,
            0.05 * (target_twist[:3] - predicted_twist[:3]) / 0.01,
            0.05 * (target_twist[3:] - predicted_twist[3:]) / 0.05,
        ]
        return 0.5 * float(residual @ residual)

    command = result["command"]
    numeric = np.array(
        [
            (
                independent_objective(command + np.eye(6)[axis] * 1e-5)
                - independent_objective(command - np.eye(6)[axis] * 1e-5)
            )
            / 2e-5
            for axis in range(6)
        ]
    )
    np.testing.assert_allclose(numeric, result["gradient"], atol=2e-6, rtol=1e-6)
    assert np.max(np.abs(projected_gradient(numeric, command, result["lower"], result["upper"]))) < 4e-6


def test_ambiguous_pi_error_and_infeasible_previous_are_rejected():
    c, model, interface = setup()
    desired = np.eye(4)
    desired[:3, :3] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    with pytest.raises(ValueError, match="pi"):
        task_objective(model, np.zeros(6), interface, np.eye(4), desired, np.zeros(6), np.zeros(6), np.zeros(12))
    with pytest.raises(ValueError, match="previous"):
        solve_pose(
            model, np.eye(4), np.eye(4), np.zeros(6), np.zeros(6), np.ones(6) * 2, np.eye(3), c, innovation=np.zeros(12)
        )


def test_json_modes_are_independently_verified_and_corruptions_rejected():
    c, model, _ = setup()
    measured, desired = np.eye(4), np.eye(4)
    desired[:3, 3] = [0.002, -0.001, 0.003]
    desired[:3, :3] = Rotation.from_rotvec([0.02, -0.03, 0.01]).as_matrix()
    previous, twist, reference, bias = np.zeros(6), np.arange(6) * 0.01, np.zeros(6), np.zeros(12)
    previous_goal = Rotation.from_rotvec([0.1, -0.05, 0.04]).as_matrix()
    result = solve_pose(model, measured, desired, twist, reference, previous, previous_goal, c, innovation=bias)
    saved = json.loads(json.dumps(result, default=lambda value: value.tolist(), allow_nan=False))
    arguments = (
        model.base["mass"],
        model.base["jacobian"],
        c,
        measured,
        desired,
        twist,
        reference,
        previous,
        previous_goal,
        bias,
    )
    certificate = verify_pose_solution(saved, *arguments)
    assert len(certificate["mode_certificates"]) == 2 and certificate["max_independent_stationarity"] < 2e-6
    for path in (
        ("command",),
        ("objective",),
        ("predicted_pose",),
        ("innovation",),
        ("mode_candidates", 0, "nominal_response"),
        ("mode_candidates", 0, "gradient"),
    ):
        broken = copy.deepcopy(saved)
        target = broken
        for key in path[:-1]:
            target = target[key]
        value = np.asarray(target[path[-1]], dtype=float).copy()
        value.flat[0] += 0.02
        target[path[-1]] = value.tolist()
        with pytest.raises(AssertionError):
            verify_pose_solution(broken, *arguments)
