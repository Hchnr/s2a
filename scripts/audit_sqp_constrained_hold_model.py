"""Frozen local hold rollout: distinguish box/rate effects without replaying future states."""

import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from scipy.spatial.transform import Rotation

from s2a.current_coupled_response import build_response
from s2a.durable_stage_supervisor import publish
from s2a.strict_convex_qp import solve_qp


def rollout(a, b, weight, state, previous, lo, hi, *, rate_limit):
    weighted_b = weight[:, None] * b
    hessian = weighted_b.T @ weighted_b
    constraints = np.vstack((np.eye(6), -np.eye(6)))
    states, commands, certificates = [state.copy()], [], []
    limited = saturated = 0
    for _ in range(200):
        lower = np.maximum(lo, previous - 0.25) if rate_limit else lo
        upper = np.minimum(hi, previous + 0.25) if rate_limit else hi
        linear = weighted_b.T @ (weight * (a @ state))
        result = solve_qp(hessian, linear, constraints, np.r_[upper, -lower], previous)
        command = np.clip(result["solution"], lower, upper)
        gradient = hessian @ command + linear
        projected = gradient.copy()
        at_lower = np.isclose(command, lower, atol=1e-9, rtol=0)
        at_upper = np.isclose(command, upper, atol=1e-9, rtol=0)
        projected[at_lower] = np.minimum(projected[at_lower], 0)
        projected[at_upper] = np.maximum(projected[at_upper], 0)
        certificate = float(np.max(np.abs(projected)))
        assert certificate < 2e-6
        assert np.all(command >= lo - 1e-10) and np.all(command <= hi + 1e-10)
        if rate_limit:
            assert np.max(np.abs(command - previous)) <= 0.25 + 1e-10
        limited += bool(np.any(np.isclose(np.abs(command - previous), 0.25, atol=1e-9, rtol=0)))
        saturated += bool(
            np.any(np.isclose(command, lo, atol=1e-9, rtol=0)) or np.any(np.isclose(command, hi, atol=1e-9, rtol=0))
        )
        # Every subsequent state comes from the selected command and this local model.
        state = a @ state + b @ command
        states.append(state.copy())
        commands.append(command.copy())
        certificates.append(certificate)
        previous = command
    states = np.asarray(states)
    return dict(
        rate_limit=rate_limit,
        model_periods=200,
        states=states.tolist(),
        commands=np.asarray(commands).tolist(),
        maximum_projected_gradient=max(certificates),
        position_max_m=float(np.linalg.norm(states[:, :3], axis=1).max()),
        rotation_tangent_max_rad=float(np.linalg.norm(states[:, 3:6], axis=1).max()),
        final_position_error_m=float(np.linalg.norm(states[-1, :3])),
        final_rotation_tangent_error_rad=float(np.linalg.norm(states[-1, 3:6])),
        tail40_position_rmse_m=float(np.sqrt(np.mean(np.sum(states[-40:, :3] ** 2, axis=1)))),
        tail40_rotation_tangent_rmse_rad=float(np.sqrt(np.mean(np.sum(states[-40:, 3:6] ** 2, axis=1)))),
        rate_boundary_periods=limited,
        saturated_periods=saturated,
    )


def main():
    output = Path("outputs/full_coupled_sqp_hold_model_20261026/verified.json")
    assert not output.exists()
    e = Evidence()
    for path in (__file__, "src/s2a/current_coupled_response.py", "src/s2a/strict_convex_qp.py"):
        e.read(path)
    proof = e.json("outputs/full_coupled_sqp_case00_tracking_20261026/verified.json")
    assert proof["complete"] and proof["reference_values_equal"]
    results = []
    for case_index, index in ((0, 120), (1, 200), (2, 140)):
        root = Path("outputs/full_coupled_sqp_control_20261026") / f"case{case_index:02d}"
        verified = e.json(root / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        case = next(root.glob("simulation/*/*"))
        rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
        row = rows[index]
        controller = e.json(case / "controller.json")
        contract = controller["contract"]
        actual = row["coupled_response_model"]
        local = build_response(actual["mass"], actual["jacobian"], actual["kp"], actual["kd"], np.eye(3))
        lo, hi = np.asarray(contract["input_min"]), np.asarray(contract["input_max"])
        outlo, outhi = np.asarray(contract["output_min"]), np.asarray(contract["output_max"])
        scale = (outhi - outlo) / (hi - lo)
        np.testing.assert_allclose(outlo - lo * scale, 0, atol=1e-14, rtol=0)
        a = np.zeros((12, 12))
        a[:6, :6] = np.eye(6)
        a[:, 6:] = local["initial_velocity_response"]
        b = local["goal_forcing_response"] * scale[None, :]
        weight = np.array([100.0] * 3 + [20.0] * 3 + [5.0] * 3 + [1.0] * 3)
        current = np.asarray(row["measured_source"]) @ np.asarray(controller["source_from_site"])
        target = np.asarray(row["desired_site"])
        state = np.r_[
            current[:3, 3] - target[:3, 3],
            Rotation.from_matrix(current[:3, :3] @ target[:3, :3].T).as_rotvec(),
            row["measured_twist"],
        ]
        previous = np.asarray(rows[index - 1]["command"][:6])
        gain = np.linalg.lstsq(weight[:, None] * b, weight[:, None] * a, rcond=None)[0]
        closed_loop = a - b @ gain
        outcomes = [
            rollout(a, b, weight, state.copy(), previous.copy(), lo, hi, rate_limit=mode) for mode in (True, False)
        ]
        results.append(
            dict(
                case=case_index,
                initial_step=index,
                initial_state=state.tolist(),
                previous_command=previous.tolist(),
                frozen_mass=local["mass"].tolist(),
                frozen_jacobian=local["jacobian"].tolist(),
                a=a.tolist(),
                b=b.tolist(),
                weight=weight.tolist(),
                unconstrained_hold_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(closed_loop)))),
                weighted_input_condition=float(np.linalg.cond(weight[:, None] * b)),
                outcomes=outcomes,
            )
        )
    e.recheck()
    publish(
        output,
        dict(
            complete=True,
            scope="frozen M/J small-angle constant-reference local model;200 propagated periods;no future measured feedback",
            assumptions="identity relative-goal tangent, zero innovation, fixed M/J, no contact/object state, H1 quadratic objective",
            physical_causal_explanation_proven=False,
            results=results,
            source_sha256=e.hashes,
            seed=42,
            new_physics_steps=0,
            model_calls=0,
            fits=0,
            fresh_cases=0,
        ),
    )
    for result in results:
        print("case", result["case"], "spectral radius", result["unconstrained_hold_spectral_radius"])
        for outcome in result["outcomes"]:
            print({key: value for key, value in outcome.items() if key not in ("states", "commands")})


if __name__ == "__main__":
    main()
