"""Replay the exact383-period prefix and compare one actual native action."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_reentry_native_20261028"
    require(not root.exists(), "No overwrite/retry")
    local_ref = reference(REPO / "outputs/tracking_v5_buffer_reentry_v2_20261028/result.json")
    local = read(local_ref)
    require(local["complete"] is local["passed"] is True, "Closed local search")
    capture = read(local["registration_ref"])
    worker = read(capture["worker_ref"])
    folder = Path(worker["physical_case_dir"])
    manifest = read(capture["manifest_ref"])
    rows = [json.loads(x) for x in checked(capture["rows_ref"]).read_text().splitlines()]
    receipts = [json.loads(x) for x in checked(worker["endpoint_receipts_ref"]).read_text().splitlines()]
    entry = next(e for e in local["entries"] if e["step"] == 383)
    require(entry["candidate"]["witness_found"] is True, "Actual reentry witness")
    candidate = np.asarray(rows[383]["command"]).copy()
    candidate[:6] = entry["candidate"]["planned_commands"][0]
    contract = read(reference(folder / "controller.json"))["contract"]
    require(
        np.all(candidate[:6] >= contract["input_min"]) and np.all(candidate[:6] <= contract["input_max"]),
        "Original box",
    )
    require(np.max(abs(candidate[:6] - np.asarray(rows[382]["command"])[:6])) <= 0.25 + 1e-12, "Original rate")
    require(not np.array_equal(candidate, rows[383]["command"]), "Actual command intervention")
    registration = publish(
        root / "registration.json",
        dict(
            local_ref=local_ref,
            capture_registration_ref=local["registration_ref"],
            worker_ref=capture["worker_ref"],
            receipts_ref=worker["endpoint_receipts_ref"],
            source_refs=[reference(Path(__file__))],
            intervention_step=383,
            candidate_command=candidate,
            physics_budget=768,
            world_model_calls=0,
            fresh_cases=0,
            seed=42,
            automatic_retry=False,
            scope="Two independently initialized383-prefix + one-period branches, recorded gripper, no new WAM",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.geometry import pose_error
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import current_observation, make_environment, source_pose

    env = None
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        branches=[],
    )
    try:
        tool = np.asarray(read(local["tool_ref"])["source_from_tcp"])
        for branch in ("baseline", "candidate"):
            env, _ = make_environment(manifest["runtime_config"], root / branch / "native", manifest["record"], 0)
            robot = env.robots[0]
            robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
            bounds = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            states, commands = [], []
            last_receipt = None
            for i in range(385):
                sample, receipt = endpoint(env, i)
                states.append(env.sim.get_state().flatten().copy())
                if i <= 383 or branch == "baseline":
                    require(
                        receipt["sim_state_sha256"] == receipts[i]["sim_state_sha256"],
                        f"Exact original state {branch}/{i}",
                    )
                if i in (0, 16, 56):
                    render_anchor(env, i, sample, receipt, root / branch)
                if i == 384:
                    last_receipt = receipt
                    break
                if i % 8 == 0:
                    render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
                command = candidate.copy() if branch == "candidate" and i == 383 else np.asarray(rows[i]["command"])
                commands.append(command.copy())
                env.step(command)
                result["physics_steps"] += 1
                current_observation(env)
            obs = current_observation(env)
            q = env.sim.data.qpos[np.asarray(robot._ref_joint_pos_indexes)].copy()
            margin = np.minimum(q - bounds[:, 0], bounds[:, 1] - q)
            path = root / branch / "states.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, sim_states=np.asarray(states), commands=np.asarray(commands))
            result["branches"].append(
                dict(
                    branch=branch,
                    states_ref=reference(path),
                    prefix_exact_states=384,
                    endpoint_receipt=last_receipt,
                    final_joint_position=q,
                    final_joint_margin_rad=margin,
                    minimum_final_margin_rad=float(margin.min()),
                    tcp_error=pose_error(source_pose(obs) @ tool, np.asarray(rows[383]["desired_tcp"])),
                    native_success=bool(env.check_success()),
                    physics_steps=384,
                )
            )
            env.close()
            env = None
        require(result["physics_steps"] == 768, "Exact paired physics budget")
        result.update(
            complete=True,
            passed=True,
            scope="One physical endpoint comparison; no inner2ms margin proof, multi-period recovery, task or WAM qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
