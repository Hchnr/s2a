"""223 same-state horizon/SOURCE comparison and measured grasp divergence."""

import json

import numpy as np
from audit_wam_position import Evidence
from executed_residual_independent import ExecutedResidualReplay
from pose_velocity_arm_independent_replay import PoseVelocityArmReplay
from serialized_rate_preview_independent import verify_plan

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.osc import OSCInterface
from s2a.rate_constrained_preview_active_set import plan_translation


def load_case(e, root, index):
    directory = root / f"case{index:02d}"
    verified = e.json(directory / "independent_verified.json")
    assert verified["complete"] and verified["mechanical_labels_images_passed"]
    case = next((directory / "simulation").glob("*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    return (
        rows,
        e.json(case / "controller.json"),
        e.json(case / "manifest.json"),
        e.json(directory / "execution_config.json"),
    )


def first(rows, predicate):
    return next((row["step"] for row in rows if predicate(row)), None)


def main():
    root = REPO / "outputs/preview_remaining_failures_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "scripts/executed_residual_independent.py",
        "scripts/pose_velocity_arm_independent_replay.py",
        "scripts/serialized_rate_preview_independent.py",
        "scripts/rate_preview_independent.py",
        "src/s2a/rate_constrained_preview_active_set.py",
        "src/s2a/strict_convex_qp.py",
        "plans/v0.0.1/223_preview_remaining_failure_audit_20261026.md",
    ):
        e.read(path)
    preview = REPO / "outputs/preview_control_20261026"
    baseline = REPO / "outputs/causal_residual_loaded_control_20261025"
    for path in (
        REPO / "outputs/preview_control_serialized_20261026/closure_verified.json",
        baseline / "closure_verified.json",
    ):
        closure = e.json(path)
        assert closure["complete"] and closure["owner_normal_wait_verified"]
        for source, signature in closure["source_sha256"].items():
            e.check(source, signature)
    registration = e.json("configs/preview_control_registered_20261026.json")
    base_registration = e.json("configs/causal_residual_loaded_control_20261025.json")
    assert registration["profiles"] == base_registration["profiles"]
    cases = []
    for index in (3, 6, 9, 10, 12, 14, 15):
        rows, controller, manifest, config = load_case(e, preview, index)
        old, old_controller, old_manifest, old_config = load_case(e, baseline, index)
        assert len(rows) == len(old) == registration["profiles"][index]["physics_steps"]
        assert controller["contract"] == old_controller["contract"]
        assert manifest["record"]["initial_state_sha256"] == old_manifest["record"]["initial_state_sha256"]
        assert config["stable_success_last_steps"] == old_config["stable_success_last_steps"]
        contract = controller["contract"]
        interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
        source = ExecutedResidualReplay(
            interface, contract["kp"], contract["kd"], 0.05, compensate=True, inverse_weight=0.5, max_command_rate=5
        )
        physical = PoseVelocityArmReplay(interface, contract["kp"], contract["kd"], 0.05)
        lo, hi, outlo, outhi = (
            np.asarray(contract[key])[:3] for key in ("input_min", "input_max", "output_min", "output_max")
        )
        observations = []
        for step, (row, prior) in enumerate(zip(rows, old, strict=True)):
            np.testing.assert_allclose(row["desired_source"], prior["desired_source"], atol=1e-9, rtol=0)
            data = {key: np.asarray(value) for key, value in row["qualified_forward_input"].items()}
            command = np.asarray(row["command"])
            measured, desired, twist = (
                np.asarray(row[key]) for key in ("measured_source", "desired_source", "measured_twist")
            )
            reconstructed = source.convert(
                measured, desired, twist, desired_linear_velocity=row["linear_velocity_reference"]["velocity"]
            )
            for key, value in reconstructed["response_residual"].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_allclose(row["physical_response_residual"][key], value, atol=1e-9, rtol=0)
                else:
                    assert row["physical_response_residual"][key] == value
            committed = source.commit_executed(measured, twist, command[:6])
            for key, value in committed.items():
                np.testing.assert_allclose(row["physical_response_commit"][key], value, atol=1e-9, rtol=0)
            position, velocity = data["x_site_pose"][0, :3, 3], data["x_site_twist"][0, :3]
            target = np.asarray(row["desired_site"])[None, :3, 3]
            target_velocity = np.asarray(row["site_velocity_reference"]["velocity"])[None]
            previous = data["x_previous_command"][0, :3]
            innovation = np.vstack([data["x_response_residual"][0, :3], data["x_response_residual"][0, 6:9] / 0.05])
            h1 = plan_translation(
                position, velocity, target, target_velocity, previous, contract, innovation=innovation
            )
            certificate = verify_plan(h1, position, velocity, target, target_velocity, previous, contract, innovation)
            r, b = physical.response[:3], physical.momentum[:3]
            c, d = physical.kp[:3] * b, 1 - r - physical.kd[:3] * b
            delta = (
                r * (target[0] - position - b * velocity - innovation[0])
                + 0.05**2 * c * (target_velocity[0] - d * velocity - innovation[1])
            ) / (r * r + 0.05**2 * c * c)
            analytic = lo + (delta - outlo) * (hi - lo) / (outhi - outlo)
            analytic = np.minimum(
                np.minimum(hi, previous + 0.25), np.maximum(np.maximum(lo, previous - 0.25), analytic)
            )
            np.testing.assert_allclose(h1["command"], analytic, atol=1e-9, rtol=0)
            after_site = np.asarray(row["after_source"]) @ np.asarray(controller["source_from_site"])
            disagreements = {
                name: float(np.linalg.norm(np.asarray(obj["position"]) - prior["objects"][name]["position"]))
                for name, obj in row["objects"].items()
            }
            fingers = np.asarray(row["gripper_qpos_after"])
            old_fingers = np.asarray(prior["gripper_qpos_after"])
            motor = np.asarray(row["motor_target_before"])
            old_motor = np.asarray(prior["motor_target_before"])
            observations.append(
                dict(
                    step=step,
                    actual_position_error_m=float(np.linalg.norm(row["tcp_error"][:3])),
                    prior085_position_error_m=float(np.linalg.norm(prior["tcp_error"][:3])),
                    h6_vs_h1_command_norm=float(np.linalg.norm(command[:3] - h1["command"])),
                    h1_vs_085_same_state_command_norm=float(
                        np.linalg.norm(h1["command"] - reconstructed["command"][:3])
                    ),
                    h1_command=h1["command"].tolist(),
                    source085_candidate=reconstructed["command"][:3].tolist(),
                    actual_command=command[:3].tolist(),
                    h1_kkt_residual=certificate["max_stationarity_residual"],
                    rate_limited=bool(row["rate_limited"]),
                    preview_prediction_error_m=float(
                        np.linalg.norm(
                            after_site[:3, 3] - np.asarray(row["translation_preview"]["predicted_positions"])[0]
                        )
                    ),
                    actual_nominal_innovation_after_m=float(
                        np.linalg.norm(after_site[:3, 3] - position - np.asarray(row["nominal_predicted_motion"])[:3])
                    ),
                    completed_position_innovation_m=float(np.linalg.norm(innovation[0])),
                    finger_gap_after_m=float(fingers[0] - fingers[1]),
                    prior085_finger_gap_after_m=float(old_fingers[0] - old_fingers[1]),
                    finger_gap_disagreement_after_m=float(
                        abs((fingers[0] - fingers[1]) - (old_fingers[0] - old_fingers[1]))
                    ),
                    motor_target_gap_before_m=float(0.04 + 0.02 * (motor[0] - motor[1])),
                    prior085_motor_target_gap_before_m=float(0.04 + 0.02 * (old_motor[0] - old_motor[1])),
                    gripper_command=row["command"][-1],
                    prior085_gripper_command=prior["command"][-1],
                    route=row["gripper"]["route"],
                    prior085_route=prior["gripper"]["route"],
                    current_force_norm=float(np.linalg.norm(data["x_current_wrench_world"][0, :3])),
                    success=row["success"],
                    prior085_success=prior["success"],
                    object_disagreements_after_m=disagreements,
                    max_object_disagreement_after_m=max(disagreements.values()),
                )
            )
        summary = dict(
            case=index,
            rows=len(rows),
            command_difference_horizon_rmse=float(
                np.sqrt(np.mean([row["h6_vs_h1_command_norm"] ** 2 for row in observations]))
            ),
            command_difference_frame_rmse=float(
                np.sqrt(np.mean([row["h1_vs_085_same_state_command_norm"] ** 2 for row in observations]))
            ),
            max_h1_kkt_residual=max(row["h1_kkt_residual"] for row in observations),
            first_object_disagreement_1cm_after_step=first(
                observations, lambda row: row["max_object_disagreement_after_m"] > 0.01
            ),
            first_finger_gap_disagreement_5mm_after_step=first(
                observations, lambda row: row["finger_gap_disagreement_after_m"] > 0.005
            ),
            first_gripper_command_disagreement_step=first(
                observations, lambda row: row["gripper_command"] != row["prior085_gripper_command"]
            ),
            first_success_step=first(observations, lambda row: row["success"]),
            prior085_first_success_step=first(observations, lambda row: row["prior085_success"]),
            peak_actual_error_step=max(observations, key=lambda row: row["actual_position_error_m"])["step"],
        )
        cases.append(dict(summary=summary, observations=observations))
    e.recheck()
    root.mkdir()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            cases=cases,
            source_sha256=e.hashes,
            physics_steps=0,
            new_fits=0,
            model_calls=0,
            seed=42,
            evidence_level="matched executed traces plus counterfactual commands on recorded states;no alternate dynamics",
            unique_physical_cause_proven=False,
            control_qualified=False,
        ),
    )
    print(json.dumps([case["summary"] for case in cases]))


if __name__ == "__main__":
    main()
