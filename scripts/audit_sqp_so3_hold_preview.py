"""Fixed preview feedback against a goal-dependent SO3 local plant; no native qualification."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from full_coupled_pose_independent import objective as independent_objective
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation

from s2a.durable_stage_supervisor import publish
from s2a.full_coupled_pose_inverse import FullCoupledPoseModel
from s2a.osc import OSCInterface
from s2a.strict_convex_qp import solve_qp


def so3_rollout(record, horizon, contract, measured, desired, previous_goal):
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
        # Plant updates compose SO3 motions and reevaluate the actual goal-local spring.
        # Float64 reset/latch semantics only; native float32 decoding is not certified here.
        command = np.clip(command, interface.input_min, interface.input_max)
        delta = interface.decode(command)
        mode = "latch" if np.all(delta[3:] == 0) else "reset"
        relative = previous_goal @ measured[:3, :3].T if mode == "latch" else None
        nominal = model.response(delta, twist, fixed_relative_goal=relative)
        checked = independent_objective(
            command,
            record["frozen_mass"],
            record["frozen_jacobian"],
            contract,
            measured,
            desired,
            twist,
            np.zeros(6),
            np.zeros(12),
            previous_goal,
            mode,
        )
        maximum_independent_response_error = max(
            maximum_independent_response_error, float(np.max(np.abs(nominal - checked["nominal_response"])))
        )
        np.testing.assert_allclose(nominal, checked["nominal_response"], atol=1e-8, rtol=0)
        if mode == "reset":
            previous_goal = Rotation.from_rotvec(delta[3:]).as_matrix() @ measured[:3, :3]
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

    output = Path("outputs/full_coupled_sqp_so3_hold_preview_20261026/verified.json")
    assert not output.exists()
    evidence = Evidence()
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
            outcome = so3_rollout(record, horizon, contract, measured, desired, previous_goal)
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
            scope="frozen M/J goal-dependent SO3 local plant; float64 feedback only",
            horizons=[1, 4],
            periods_per_rollout=200,
            results=results,
            native_decode_qualified=False,
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
