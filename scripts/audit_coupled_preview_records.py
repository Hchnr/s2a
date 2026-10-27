"""Conditional preview proposals on completed records; never a counterfactual rollout."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from coupled_pose_preview_independent import verify_preview
from current_coupled_independent import response
from full_coupled_arm_log_replay import innovation
from full_coupled_pose_independent import generated_twist

from s2a.action_reference import decode_action_targets
from s2a.coupled_pose_preview import plan_pose
from s2a.durable_stage_supervisor import publish
from s2a.generated_reference_twist import reference_world_twist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.cases <= 18 and not args.output.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/coupled_pose_preview.py",
        "scripts/coupled_pose_preview_independent.py",
        "src/s2a/current_coupled_response.py",
        "scripts/current_coupled_independent.py",
        "src/s2a/strict_convex_qp.py",
        "src/s2a/action_reference.py",
        "src/s2a/geometry.py",
        "src/s2a/generated_reference_twist.py",
        "src/s2a/reference_linear_velocity.py",
        "scripts/full_coupled_arm_log_replay.py",
        "scripts/full_coupled_pose_independent.py",
    ):
        e.read(path)
    cases = []
    for case_index in range(args.cases):
        root = args.root / f"case{case_index:02d}"
        prior = e.json(root / "independent_verified.json")
        assert prior["complete"] and prior["mechanical_labels_images_passed"]
        case = next(root.glob("simulation/*/*"))
        controller = e.json(case / "controller.json")
        contract = controller["contract"]
        manifest = e.json(case / "manifest.json")
        tool = e.json(manifest["runtime_config"]["tool_contract"])
        tcp_to_site = np.linalg.inv(np.asarray(tool["source_from_tcp"])) @ np.asarray(controller["source_from_site"])
        chunks = e.json(case / "chunks.json")
        rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
        decoded = []
        for chunk in chunks:
            poses, _ = decode_action_targets(
                chunk["anchor_tcp"], chunk["effective128"], horizon=len(chunk["effective128"])
            )
            np.testing.assert_allclose(poses, chunk["effective_targets_tcp"], atol=1e-9, rtol=0)
            decoded.append((poses @ tcp_to_site, np.asarray(chunk["anchor_tcp"]) @ tcp_to_site))
        previous = np.zeros(6)
        last = None
        checks = []
        for index, row in enumerate(rows):
            assert row["step"] == index
            pose = np.asarray(row["measured_source"]) @ np.asarray(controller["source_from_site"])
            twist = np.asarray(row["measured_twist"])
            bias = innovation(last, pose, twist, scaled_velocity=False)
            np.testing.assert_allclose(bias, row["coupled_response_history"]["innovation_before"], atol=1e-8, rtol=0)
            assert row["coupled_response_history"]["completed_periods_before"] == index
            packet = row["qualified_forward_input"]
            np.testing.assert_allclose(packet["x_previous_command"][0][:6], previous, atol=1e-12, rtol=0)
            chunk_index = row["chunk_index"]
            poses, anchor = decoded[chunk_index]
            current = int(row["site_velocity_reference"]["current_index"])
            assert chunks[chunk_index]["step"] <= index
            np.testing.assert_allclose(poses[current], row["desired_site"], atol=1e-8, rtol=0)
            end = min(current + 4, len(poses))
            references = poses[current:end]
            twists = []
            for target_index in range(current, end):
                actual = reference_world_twist(poses, target_index, anchor, 0.05)["twist"]
                independent = generated_twist(poses, target_index, anchor, 0.05)["twist"]
                np.testing.assert_allclose(actual, independent, atol=1e-10, rtol=0)
                twists.append(actual)
            dynamics = row["coupled_response_model"]
            parameters = dict(
                measured=pose,
                twist=twist,
                references=references,
                reference_twists=np.asarray(twists),
                previous_command=previous,
                mass=dynamics["mass"],
                jacobian=dynamics["jacobian"],
                contract=contract,
                innovation=bias,
            )
            planned = plan_pose(**parameters)
            certificate = verify_preview(planned, **parameters)
            checks.append(
                dict(
                    step=index,
                    chunk=chunk_index,
                    current_index=current,
                    horizon=len(references),
                    command=planned["command"].tolist(),
                    certificate=certificate,
                )
            )
            # Advance history using the recorded actual command, never this unexecuted proposal.
            actual_command = np.asarray(row["command"])[:6]
            lo, hi, outlo, outhi = (
                np.asarray(contract[k]) for k in ("input_min", "input_max", "output_min", "output_max")
            )
            delta = (actual_command - (hi + lo) / 2) * (outhi - outlo) / (hi - lo) + (outhi + outlo) / 2
            native_relative = np.asarray(row["controller_goal_after"]) @ pose[:3, :3].T
            independent = response(
                dynamics["mass"], dynamics["jacobian"], contract["kp"], contract["kd"], native_relative
            )
            nominal = (
                independent["initial_velocity_response"] @ twist
                + independent["goal_forcing_response"] @ np.r_[delta[:3], independent["rotation_forcing"]]
            )
            np.testing.assert_allclose(
                nominal, row["coupled_response_history"]["commit"]["nominal_response"], atol=1e-8, rtol=0
            )
            last = (pose.copy(), nominal)
            previous = actual_command.copy()
        cases.append(dict(case=case_index, periods=len(checks), checks=checks))
        print(
            dict(
                case=case_index,
                periods=len(checks),
                maximum_kkt=max(x["certificate"]["max_independent_stationarity"] for x in checks),
            ),
            flush=True,
        )
    e.recheck()
    publish(
        args.output,
        dict(
            complete=True,
            scope="conditional proposals on recorded states and executed histories; no counterfactual control",
            cases=cases,
            source_sha256=e.hashes,
            seed=42,
            new_physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
