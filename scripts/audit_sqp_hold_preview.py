"""Compare fixed preview horizons under unchanged local hold box/rate constraints."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from scipy.optimize import nnls

from s2a.durable_stage_supervisor import publish
from s2a.strict_convex_qp import solve_qp


def preview_rollout(record, horizon):
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
    states, commands, residuals = [state.copy()], [], []
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
        state = a @ state + b @ command
        states.append(state.copy())
        commands.append(command.copy())
        residuals.append(residual)
        previous = command.copy()
        warm = np.r_[sequence[6:], sequence[-6:]]
    states = np.asarray(states)
    return dict(
        horizon=horizon,
        states=states.tolist(),
        commands=np.asarray(commands).tolist(),
        maximum_independent_kkt=max(residuals),
        final_position_error_m=float(np.linalg.norm(states[-1, :3])),
        final_rotation_tangent_error_rad=float(np.linalg.norm(states[-1, 3:6])),
        tail40_position_rmse_m=float(np.sqrt(np.mean(np.sum(states[-40:, :3] ** 2, axis=1)))),
        tail40_rotation_tangent_rmse_rad=float(np.sqrt(np.mean(np.sum(states[-40:, 3:6] ** 2, axis=1)))),
    )


def main():
    output = Path("outputs/full_coupled_sqp_hold_preview_20261026/verified.json")
    assert not output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.read("src/s2a/strict_convex_qp.py")
    source = evidence.json("outputs/full_coupled_sqp_hold_model_20261026/verified.json")
    assert source["complete"] and not source["physical_causal_explanation_proven"]
    for path, digest in source["source_sha256"].items():
        evidence.check(path, digest)
    results = []
    for record in source["results"]:
        root = Path("outputs/full_coupled_sqp_control_20261026") / f"case{record['case']:02d}"
        case = next(root.glob("simulation/*/*"))
        contract = evidence.json(case / "controller.json")["contract"]
        np.testing.assert_array_equal(contract["input_min"], [-1] * 6)
        np.testing.assert_array_equal(contract["input_max"], [1] * 6)
        for horizon in (1, 4, 8):
            outcome = preview_rollout(record, horizon)
            if horizon == 1:
                np.testing.assert_allclose(outcome["states"], record["outcomes"][0]["states"], atol=1e-9, rtol=1e-8)
            outcome.update(case=record["case"], initial_step=record["initial_step"])
            results.append(outcome)
            print({k: v for k, v in outcome.items() if k not in ("states", "commands")}, flush=True)
    evidence.recheck()
    publish(
        output,
        dict(
            complete=True,
            seed=42,
            scope="fixed M/J small-angle local model only; unchanged box and rate bounds",
            horizons=[1, 4, 8],
            periods_per_rollout=200,
            results=results,
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
