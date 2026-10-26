"""Local preview with actual native command scaling and set_goal; no simulator stepping."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from current_coupled_independent import response as independent_response
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation

from s2a.durable_stage_supervisor import publish
from s2a.full_coupled_pose_inverse import FullCoupledPoseModel
from s2a.osc import OSCInterface
from s2a.strict_convex_qp import solve_qp


def native_rollout(record, horizon, contract, measured, desired, previous_goal):
    from robosuite.controllers.osc import OperationalSpaceController
    from robosuite.utils import transform_utils as transform

    a, b, weight = (np.asarray(record[key]) for key in ("a", "b", "weight"))
    state, previous = (np.asarray(record[key]) for key in ("initial_state", "previous_command"))
    prediction = np.vstack([np.linalg.matrix_power(a, k + 1) for k in range(horizon)])
    response = np.zeros((12 * horizon, 6 * horizon))
    for k in range(horizon):
        for j in range(k + 1):
            response[12 * k : 12 * (k + 1), 6 * j : 6 * (j + 1)] = np.linalg.matrix_power(a, k - j) @ b
    weights = np.tile(weight, horizon)
    weighted = weights[:, None] * response
    hessian = weighted.T @ weighted
    difference = np.eye(6 * horizon)
    for k in range(1, horizon):
        difference[6 * k : 6 * (k + 1), 6 * (k - 1) : 6 * k] = -np.eye(6)
    eye = np.eye(6 * horizon)
    constraints = np.vstack((eye, -eye, difference, -difference))
    model = FullCoupledPoseModel(record["frozen_mass"], record["frozen_jacobian"], contract["kp"], contract["kd"])
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    measured, desired, previous_goal = measured.copy(), desired.copy(), previous_goal.copy()
    twist = state[6:].copy()
    # Isolated native goal updater: endpoint refresh supplied by the local model.
    controller = OperationalSpaceController.__new__(OperationalSpaceController)
    controller.update = lambda: None
    controller.impedance_mode = "fixed"
    controller.use_delta = controller.use_ori = True
    controller.position_limits = controller.orientation_limits = None
    controller.interpolator_pos = controller.interpolator_ori = None
    controller.action_scale = None
    for key in ("input_min", "input_max", "output_min", "output_max"):
        setattr(controller, key, np.asarray(contract[key]))
    controller.goal_ori = previous_goal.copy()
    numeric_mode_differences = 0
    states, commands, residuals = [state.copy()], [], []
    modes, maximum_independent_response_error = [], 0.0
    warm = np.tile(previous, horizon)
    for _ in range(200):
        center = np.r_[previous, np.zeros(6 * (horizon - 1))]
        bounds = np.r_[np.ones(12 * horizon), 0.25 + center, 0.25 - center]
        linear = weighted.T @ (weights * (prediction @ state))
        solved = solve_qp(hessian, linear, constraints, bounds, warm)
        sequence = solved["solution"]
        # Independent multiplier reconstruction, not the solver's certificate.
        slack = bounds - constraints @ sequence
        active = constraints[slack < 1e-8]
        gradient = hessian @ sequence + linear
        multipliers = nnls(active.T, -gradient, maxiter=10000)[0] if len(active) else np.zeros(0)
        residual = float(np.max(np.abs(gradient + active.T @ multipliers)))
        assert residual < 2e-6 and slack.min() >= -1e-9
        command = sequence[:6]
        assert np.max(np.abs(command)) <= 1 + 1e-9
        assert np.max(np.abs(command - previous)) <= 0.25 + 1e-9
        # Cross-check condensation using explicit state propagation.
        predicted = state.copy()
        explicit = []
        for future_command in sequence.reshape(horizon, 6):
            predicted = a @ predicted + b @ future_command
            explicit.append(predicted.copy())
        np.testing.assert_allclose(np.concatenate(explicit), prediction @ state + response @ sequence, atol=1e-12)
        command = np.clip(command, interface.input_min, interface.input_max)
        controller.ee_pos, controller.ee_ori_mat = measured[:3, 3].copy(), measured[:3, :3].copy()
        delta = controller.scale_action(command)
        mode = "latch" if np.all(delta[3:] == 0) else "reset"
        numeric_mode_differences += mode != ("latch" if np.all(interface.decode(command)[3:] == 0) else "reset")
        before_goal = controller.goal_ori.copy()
        controller.set_goal(command)
        # Independently reproduce centered scaling and native quaternion conversion.
        low, high, outlow, outhigh = (
            np.asarray(contract[k]) for k in ("input_min", "input_max", "output_min", "output_max")
        )
        scaled = (command - (high + low) / 2) * abs(outhigh - outlow) / abs(high - low) + (outhigh + outlow) / 2
        np.testing.assert_array_equal(delta, scaled)
        goal = (
            before_goal
            if mode == "latch"
            else transform.quat2mat(transform.axisangle2quat(scaled[3:])) @ measured[:3, :3]
        )
        np.testing.assert_array_equal(controller.goal_ori, goal)
        np.testing.assert_allclose(controller.goal_pos, measured[:3, 3] + scaled[:3], atol=1e-15, rtol=0)
        relative = controller.goal_ori @ measured[:3, :3].T
        nominal = model.response(delta, twist, fixed_relative_goal=relative)
        independent = independent_response(
            record["frozen_mass"], record["frozen_jacobian"], contract["kp"], contract["kd"], relative
        )
        expected = (
            independent["initial_velocity_response"] @ twist
            + independent["goal_forcing_response"] @ np.r_[delta[:3], independent["rotation_forcing"]]
        )
        expected_pose = measured.copy()
        expected_pose[:3, 3] += expected[:3]
        expected_pose[:3, :3] = Rotation.from_rotvec(expected[3:6]).as_matrix() @ measured[:3, :3]
        checked = dict(nominal_response=expected, predicted_pose=expected_pose)
        maximum_independent_response_error = max(
            maximum_independent_response_error, float(np.max(np.abs(nominal - checked["nominal_response"])))
        )
        np.testing.assert_allclose(nominal, checked["nominal_response"], atol=1e-8, rtol=0)
        previous_goal = controller.goal_ori.copy()
        measured[:3, 3] += nominal[:3]
        measured[:3, :3] = Rotation.from_rotvec(nominal[3:6]).as_matrix() @ measured[:3, :3]
        twist = nominal[6:].copy()
        np.testing.assert_allclose(measured, checked["predicted_pose"], atol=1e-8, rtol=0)
        state = np.r_[
            measured[:3, 3] - desired[:3, 3],
            Rotation.from_matrix(measured[:3, :3] @ desired[:3, :3].T).as_rotvec(),
            twist,
        ]
        modes.append(mode)
        states.append(state.copy())
        commands.append(command.copy())
        residuals.append(residual)
        previous = command.copy()
        warm = np.r_[sequence[6:], sequence[-6:]]
    states = np.asarray(states)
    return dict(
        horizon=horizon,
        goal_modes=modes,
        centered_vs_endpoint_mode_differences=numeric_mode_differences,
        maximum_independent_response_error=maximum_independent_response_error,
        states=states.tolist(),
        commands=np.asarray(commands).tolist(),
        maximum_independent_kkt=max(residuals),
        final_position_error_m=float(np.linalg.norm(states[-1, :3])),
        final_rotation_tangent_error_rad=float(np.linalg.norm(states[-1, 3:6])),
        tail40_position_rmse_m=float(np.sqrt(np.mean(np.sum(states[-40:, :3] ** 2, axis=1)))),
        tail40_rotation_tangent_rmse_rad=float(np.sqrt(np.mean(np.sum(states[-40:, 3:6] ** 2, axis=1)))),
    )


def main():
    import json

    output = Path("outputs/full_coupled_sqp_native_hold_preview_20261026/verified.json")
    assert not output.exists()
    evidence = Evidence()
    import inspect

    from robosuite.controllers.base_controller import Controller
    from robosuite.controllers.osc import OperationalSpaceController
    from robosuite.utils import control_utils, transform_utils

    for obj in (OperationalSpaceController, Controller, transform_utils, control_utils):
        evidence.read(inspect.getfile(obj))
    for path in (
        __file__,
        "src/s2a/strict_convex_qp.py",
        "src/s2a/full_coupled_pose_inverse.py",
        "src/s2a/current_coupled_response.py",
        "src/s2a/osc.py",
        "scripts/full_coupled_pose_independent.py",
        "scripts/current_coupled_independent.py",
    ):
        evidence.read(path)
    source = evidence.json("outputs/full_coupled_sqp_hold_model_20261026/verified.json")
    preview = evidence.json("outputs/full_coupled_sqp_hold_preview_20261026/verified.json")
    assert source["complete"] and preview["complete"]
    for proof in (source, preview):
        for path, digest in proof["source_sha256"].items():
            evidence.check(path, digest)
    results = []
    for record in source["results"]:
        root = Path("outputs/full_coupled_sqp_control_20261026") / f"case{record['case']:02d}"
        case = next(root.glob("simulation/*/*"))
        controller = evidence.json(case / "controller.json")
        contract = controller["contract"]
        rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
        row = rows[record["initial_step"]]
        measured = np.asarray(row["measured_source"]) @ np.asarray(controller["source_from_site"])
        desired = np.asarray(row["desired_site"])
        # All selected seeds have nonzero initial rotational command. The first reset
        # overwrites this placeholder; a first-step latch is rejected below.
        previous_goal = measured[:3, :3].copy()
        for horizon in (1, 4):
            outcome = native_rollout(record, horizon, contract, measured, desired, previous_goal)
            assert outcome["goal_modes"][0] == "reset", "Need real previous goal for first-step latch"
            outcome.update(case=record["case"], initial_step=record["initial_step"])
            results.append(outcome)
            print({k: v for k, v in outcome.items() if k not in ("states", "commands", "goal_modes")}, flush=True)
    evidence.recheck()
    publish(
        output,
        dict(
            complete=True,
            seed=42,
            scope="frozen M/J SO3 local plant with isolated actual native set_goal; no physics stepping",
            horizons=[1, 4],
            periods_per_rollout=200,
            results=results,
            native_goal_method_checked=True,
            physical_controller_qualified=False,
            physical_causal_explanation_proven=False,
            new_physics_steps=0,
            model_calls=0,
            fits=0,
            fresh_cases=0,
            source_sha256=evidence.hashes,
        ),
    )


if __name__ == "__main__":
    main()
