"""239 independent ODE/objective, finite-difference gradient and bound KKT replay."""

import numpy as np
from current_coupled_independent import response
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation


def generated_twist(poses, index, anchor, dt):
    poses, anchor = np.asarray(poses), np.asarray(anchor)
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4) and anchor.shape == (4, 4)
    assert type(index) is int and 0 <= index < len(poses) and dt > 0
    before = poses[index - 1] if index else anchor
    following = index + 1 < len(poses)
    after_index = index + 1 if following else index
    after = poses[after_index]
    interval = dt * (2 if following else 1)
    linear = (after[:3, 3] - before[:3, 3]) / interval
    angular = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec() / interval
    mode = (
        ("central_generated" if index else "central_query_anchor")
        if following
        else ("backward_tail" if index else "single_anchor")
    )
    return dict(
        velocity=linear,
        mode=mode,
        current_index=index,
        before_index=index - 1,
        after_index=after_index,
        has_previous_generated_reference=index > 0,
        has_next_generated_reference=following,
        twist=np.r_[linear, angular],
        angular_frame="MuJoCo world",
    )


def objective(
    command, mass, jacobian, contract, measured, desired, twist, reference_twist, innovation, previous_goal, mode
):
    command, measured, desired, twist, reference_twist, bias, previous_goal = (
        np.asarray(value, dtype=float)
        for value in (command, measured, desired, twist, reference_twist, innovation, previous_goal)
    )
    lo, hi, outlo, outhi = (
        np.asarray(contract[key], dtype=float) for key in ("input_min", "input_max", "output_min", "output_max")
    )
    delta = outlo + (command - lo) * (outhi - outlo) / (hi - lo)
    assert mode in ("reset", "latch")
    relative = Rotation.from_rotvec(delta[3:]).as_matrix() if mode == "reset" else previous_goal @ measured[:3, :3].T
    model = response(mass, jacobian, contract["kp"], contract["kd"], relative)
    nominal = (
        model["initial_velocity_response"] @ twist
        + model["goal_forcing_response"] @ np.r_[delta[:3], model["rotation_forcing"]]
    )
    pose = np.eye(4)
    pose[:3, 3] = measured[:3, 3] + nominal[:3] + bias[:3]
    pose[:3, :3] = (
        Rotation.from_rotvec(bias[3:6]).as_matrix() @ Rotation.from_rotvec(nominal[3:6]).as_matrix() @ measured[:3, :3]
    )
    predicted_twist = nominal[6:] + bias[6:] / 0.05
    error_rotation = Rotation.from_matrix(desired[:3, :3] @ pose[:3, :3].T).as_rotvec()
    residual = np.r_[
        (desired[:3, 3] - pose[:3, 3]) / 0.01,
        error_rotation / 0.05,
        0.05 * (reference_twist[:3] - predicted_twist[:3]) / 0.01,
        0.05 * (reference_twist[3:] - predicted_twist[3:]) / 0.05,
    ]
    return dict(
        objective=0.5 * float(residual @ residual),
        residual=residual,
        physical_delta=delta,
        nominal_response=nominal,
        predicted_pose=pose,
        predicted_twist=predicted_twist,
    )


def verify_pose_solution(
    saved, mass, jacobian, contract, measured, desired, twist, reference_twist, previous, previous_goal, innovation
):
    assert contract["control_freq"] == 20 and contract["control_frame"] == "MuJoCo world"
    previous = np.asarray(previous)
    lower, upper = (
        np.maximum(contract["input_min"], previous - 0.25),
        np.minimum(contract["input_max"], previous + 0.25),
    )
    assert previous.shape == (6,) and np.all(lower <= upper)
    mode_names = [item["goal_mode"] for item in saved["mode_candidates"]]
    lo, hi, outlo, outhi = (np.asarray(contract[key]) for key in ("input_min", "input_max", "output_min", "output_max"))
    zero = lo - outlo * (hi - lo) / (outhi - outlo)
    expected_modes = {"reset"}
    if np.all(zero[3:] >= lower[3:]) and np.all(zero[3:] <= upper[3:]):
        expected_modes.add("latch")
    assert len(mode_names) == len(set(mode_names)) and set(mode_names) == expected_modes
    assert set(saved["mode_objectives"]) == expected_modes
    certificates = []
    for item in saved["mode_candidates"]:
        mode = item["goal_mode"]
        command = np.asarray(item["command"])
        assert command.shape == (6,) and np.isfinite(command).all()
        assert np.all(command >= lower - 1e-12) and np.all(command <= upper + 1e-12)
        for key, value in (("lower", lower), ("upper", upper), ("innovation", innovation)):
            np.testing.assert_allclose(item[key], value, atol=1e-12, rtol=0)
        arguments = (
            mass,
            jacobian,
            contract,
            measured,
            desired,
            twist,
            reference_twist,
            innovation,
            previous_goal,
            mode,
        )
        expected = objective(command, *arguments)
        assert bool(np.all(expected["physical_delta"][3:] == 0)) == (mode == "latch")
        for key, value in expected.items():
            np.testing.assert_allclose(item[key], value, atol=1e-7 if key == "objective" else 1e-8, rtol=0)
        if mode == "latch":
            np.testing.assert_allclose(
                item["fixed_relative_goal"],
                np.asarray(previous_goal) @ np.asarray(measured)[:3, :3].T,
                atol=1e-12,
                rtol=0,
            )
        else:
            assert item["fixed_relative_goal"] is None
        assert item["native_decode_and_actual_commit_required"] and item["global_optimality_claim"] is False
        axes = np.arange(3) if mode == "latch" else np.arange(6)
        gradient = []
        for axis in axes:
            step = np.eye(6)[axis] * 2e-5
            gradient.append(
                (
                    objective(command + step, *arguments)["objective"]
                    - objective(command - step, *arguments)["objective"]
                )
                / 4e-5
            )
        gradient = np.asarray(gradient)
        np.testing.assert_allclose(np.asarray(item["gradient"])[axes], gradient, atol=2e-6, rtol=1e-6)
        matrix = np.vstack([np.eye(len(axes)), -np.eye(len(axes))])
        bound = np.r_[upper[axes], -lower[axes]]
        slack = bound - matrix @ command[axes]
        active = slack <= 1e-8
        multiplier = nnls(matrix[active].T, -gradient)[0] if np.any(active) else np.empty(0)
        stationarity = gradient + (matrix[active].T @ multiplier if np.any(active) else 0)
        residual = float(np.max(np.abs(stationarity)))
        assert residual < 2e-6
        assert item["max_projected_gradient"] <= 2e-6
        np.testing.assert_allclose(saved["mode_objectives"][mode], expected["objective"], atol=1e-7, rtol=0)
        certificates.append(
            dict(
                goal_mode=mode, independent_stationarity_residual=residual, independent_objective=expected["objective"]
            )
        )
    assert saved["goal_mode"] in mode_names
    chosen = next(item for item in saved["mode_candidates"] if item["goal_mode"] == saved["goal_mode"])
    for key in (
        "command",
        "objective",
        "gradient",
        "physical_delta",
        "nominal_response",
        "predicted_pose",
        "predicted_twist",
        "residual",
        "innovation",
        "lower",
        "upper",
    ):
        np.testing.assert_allclose(saved[key], chosen[key], atol=1e-12, rtol=0)
    assert saved["objective"] <= min(saved["mode_objectives"].values()) + 1e-12
    assert saved["native_decode_and_actual_commit_required"] and saved["global_optimality_claim"] is False
    if chosen["fixed_relative_goal"] is None:
        assert saved["fixed_relative_goal"] is None
    else:
        np.testing.assert_allclose(saved["fixed_relative_goal"], chosen["fixed_relative_goal"], atol=1e-12, rtol=0)
    return dict(
        mode_certificates=certificates,
        max_independent_stationarity=max(item["independent_stationarity_residual"] for item in certificates),
    )
