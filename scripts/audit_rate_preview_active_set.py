"""218 independent QP admission on real states and already generated references."""

import json

import numpy as np
from audit_wam_position import Evidence
from guarded_loading_acceptance_checker import independent_reference_velocity
from pose_velocity_arm_independent_replay import PoseVelocityArmReplay
from rate_preview_independent import verify_plan

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.osc import OSCInterface
from s2a.rate_constrained_preview_active_set import plan_translation


def main():
    root = REPO / "outputs/rate_preview_active_set_admission_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/rate_constrained_preview_active_set.py",
        "src/s2a/strict_convex_qp.py",
        "scripts/guarded_loading_acceptance_checker.py",
        "plans/v0.0.1/219_exact_preview_qp_20261026.md",
        "outputs/rate_preview_recorded_admission_20261026/failure.json",
        "scripts/rate_preview_independent.py",
        "scripts/pose_velocity_arm_independent_replay.py",
        "tests/test_rate_constrained_preview.py",
        "tests/test_rate_preview_active_set.py",
        "plans/v0.0.1/218_rate_constrained_preview_20261026.md",
    ):
        e.read(path)
    run = REPO / "outputs/late_clipped_inverse_control_20261026"
    checked = e.json(run / "case00/independent_verified.json")
    assert checked["complete"] and checked["mechanical_labels_images_passed"]
    case = next((run / "case00/simulation").glob("*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    assert len(rows) == 164
    chunks, controller = e.json(case / "chunks.json"), e.json(case / "controller.json")
    config = e.json(run / "case00/execution_config.json")
    template = e.json(config["policy_template"])
    tool = np.asarray(e.json(template["tool_contract"])["source_from_tcp"])
    mapping = np.linalg.inv(tool) @ np.asarray(controller["source_from_site"])
    contract = controller["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    response = PoseVelocityArmReplay(interface, contract["kp"], contract["kd"], 0.05)
    results = []
    root.mkdir()
    publish(root / "source_sha256.json", e.hashes)
    try:
        for index, row in enumerate(rows):
            chunk = chunks[row["chunk_index"]]
            current = index - chunk["step"]
            generated = np.asarray(chunk["effective_targets_tcp"]) @ mapping
            anchor = np.asarray(chunk["anchor_tcp"]) @ mapping
            np.testing.assert_allclose(generated[current], row["desired_site"], atol=1e-9, rtol=0)
            horizon = min(6, len(generated) - current)
            references = generated[current : current + horizon, :3, 3]
            velocities = np.array(
                [
                    independent_reference_velocity(generated, current + j, anchor, 0.05)["velocity"]
                    for j in range(horizon)
                ]
            )
            state = {key: np.asarray(value) for key, value in row["qualified_forward_input"].items()}
            position, velocity = state["x_site_pose"][0, :3, 3], state["x_site_twist"][0, :3]
            previous = state["x_previous_command"][0, :3]
            innovation = np.vstack([state["x_response_residual"][0, :3], state["x_response_residual"][0, 6:9] / 0.05])
            plan = plan_translation(
                position, velocity, references, velocities, previous, contract, innovation=innovation
            )
            certificate = verify_plan(plan, position, velocity, references, velocities, previous, contract, innovation)
            # Holding the actual previous command is always a feasible comparator.
            p, v = position.copy(), velocity.copy()
            physical = interface.output_min[:3] + (previous - interface.input_min[:3]) * (
                interface.output_max[:3] - interface.output_min[:3]
            ) / (interface.input_max[:3] - interface.input_min[:3])
            hold_objective = 0.0
            for j in range(horizon):
                p, v = (
                    p + response.momentum[:3] * v + response.response[:3] * physical + innovation[0],
                    (1 - response.response[:3] - response.kd[:3] * response.momentum[:3]) * v
                    + response.kp[:3] * response.momentum[:3] * physical
                    + innovation[1],
                )
                hold_objective += 0.5 * (
                    np.sum(((p - references[j]) / 0.01) ** 2) + np.sum((0.05 * (v - velocities[j]) / 0.01) ** 2)
                )
            objective = sum(axis["objective"] for axis in plan["axes"])
            assert objective <= hold_objective + 1e-8
            results.append(
                dict(
                    step=index,
                    horizon=horizon,
                    command=plan["command"].tolist(),
                    planned_commands=plan["planned_commands"].tolist(),
                    objective=objective,
                    feasible_hold_objective=float(hold_objective),
                    **certificate,
                )
            )
        e.recheck()
        publish(
            root / "independent_verified.json",
            dict(
                complete=True,
                rows=164,
                results=results,
                max_stationarity_residual=max(row["max_stationarity_residual"] for row in results),
                source_sha256=e.hashes,
                physics_steps=0,
                new_fits=0,
                model_calls=0,
                seed=42,
                evidence_level="independent convex optimum on recorded current states; no counterfactual physical execution",
                control_qualified=False,
            ),
        )
    except Exception as error:
        publish(
            root / "failure.json",
            dict(
                complete=False,
                checked_rows=len(results),
                error_type=type(error).__name__,
                error=str(error),
                source_sha256=e.hashes,
                physics_steps=0,
                automatic_retry=False,
            ),
        )
        raise
    print("218 independent optimal feasible preview verified on164states; no physical performance claim")


if __name__ == "__main__":
    main()
