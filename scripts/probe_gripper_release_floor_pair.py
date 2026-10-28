"""Paired fixed-reference native replay: original versus explicit release floor."""

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
    root = REPO / "outputs/gripper_release_floor_pair_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    worker_ref = reference(
        REPO
        / "outputs/tracking_preview_wam_development_v3_20261027/development/cases/task0_init0_s43_terminal058/worker_result.json"
    )
    worker = read(worker_ref)
    require(worker["complete"] is worker["passed"] is True, "Closed original600")
    folder = Path(worker["physical_case_dir"])
    manifest_ref, rows_ref, chunks_ref = [
        reference(folder / name) for name in ("manifest.json", "steps.jsonl", "chunks.json")
    ]
    manifest, chunks = read(manifest_ref), read(chunks_ref)
    rows = [json.loads(x) for x in checked(rows_ref).read_text().splitlines()]
    receipts = [json.loads(x) for x in checked(worker["endpoint_receipts_ref"]).read_text().splitlines()]
    require(len(rows) == 600 and len(chunks) == 75 and len(receipts) == 601, "Full fixed reference600")
    config = read(worker["config_ref"])
    parameters = dict(release_reference_minimum=0.35)
    registration = publish(
        root / "registration.json",
        dict(
            worker_ref=worker_ref,
            manifest_ref=manifest_ref,
            rows_ref=rows_ref,
            chunks_ref=chunks_ref,
            parameters=parameters,
            seed=43,
            physics_budget=1200,
            world_model_calls=0,
            fresh_cases=0,
            arm_policy="Replay identical recorded arm actions in both branches",
            reference_policy="Unchanged original terminal058 recorded chunks; original query times; not responsive WAM",
            source_refs=[reference(Path(__file__)), reference(REPO / "src/s2a/gripper_release_floor.py")],
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        branches=[],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import aperture, current_observation, make_environment, source_pose
    from s2a.loaded_velocity_control import LoadedVelocityControl
    from s2a.loaded_velocity_release_floor import LoadedVelocityControl as ReleaseFloorControl

    tool = np.asarray(read(reference(Path(manifest["runtime_config"]["tool_contract"])))["source_from_tcp"])
    env = None
    try:
        for branch in ("baseline", "candidate"):
            env, _ = make_environment(manifest["runtime_config"], root / branch / "native", manifest["record"], 0)
            robot = env.robots[0]
            obs = current_observation(env)
            robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
            control_type = LoadedVelocityControl if branch == "baseline" else ReleaseFloorControl
            grip = control_type(
                read(config["response_parameters_ref"]),
                aperture(obs),
                velocity_envelope=config["gripper_velocity_envelope"],
            )
            previous = 0.0
            successes = []
            grasp = 0
            changed = False
            log = []
            states = []
            initial_height = object_progress(env)["objects"]["akita_black_bowl_1"]["position"][2]
            max_height = 0.0
            for i in range(601):
                sample, receipt = endpoint(env, i)
                states.append(env.sim.get_state().flatten().copy())
                if branch == "baseline" or not changed:
                    require(
                        receipt["sim_state_sha256"] == receipts[i]["sim_state_sha256"], f"{branch} original prefix {i}"
                    )
                if i in (0, 16, 56):
                    render_anchor(env, i, sample, receipt, root / branch)
                if i == 600:
                    break
                if i % 8 == 0:
                    render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
                    chunk = chunks[i // 8]
                    require(chunk["step"] == i, "Fixed query delivery")
                    targets = np.asarray(chunk["effective_targets_tcp"])
                    openings = np.asarray(chunk["openings"])
                velocity = np.asarray(obs["robot0_gripper_qvel"])
                selected = grip.update(
                    aperture(obs),
                    robot.gripper.current_action,
                    0.05 * (velocity[0] - velocity[1]) / 0.079,
                    targets,
                    openings,
                    i % 8,
                    source_pose(obs) @ tool,
                    previous_command=previous,
                    dt=0.05,
                )
                command = np.asarray(rows[i]["command"]).copy()
                if branch == "baseline":
                    np.testing.assert_allclose(selected.command, command[6], atol=1e-12, rtol=0)
                else:
                    command[6] = selected.command
                    changed = changed or abs(command[6] - rows[i]["command"][6]) > 1e-12
                require(np.isfinite(command).all() and abs(command[6]) <= 1, "Finite feasible command")
                env.step(command)
                result["physics_steps"] += 1
                obs = current_observation(env)
                previous = float(command[6])
                progress = object_progress(env)["objects"]["akita_black_bowl_1"]
                grasp += int(progress["grasp_contact"])
                max_height = max(max_height, float(progress["position"][2] - initial_height))
                success = bool(env.check_success())
                successes.append(success)
                log.append(
                    dict(
                        step=i,
                        command=command.tolist(),
                        opening=float(aperture(obs)),
                        success=success,
                        grasp=bool(progress["grasp_contact"]),
                        height_gain_m=float(progress["position"][2] - initial_height),
                        gripper_route=selected.route,
                    )
                )
            states_path = root / branch / "states.npz"
            with states_path.open("xb") as stream:
                np.savez_compressed(stream, sim_states=np.asarray(states))
            branch_ref = publish(root / branch / "steps.json", log)
            result["branches"].append(
                dict(
                    branch=branch,
                    states_ref=reference(states_path),
                    steps_ref=branch_ref,
                    physics_steps=600,
                    release_reference_minimum=0.35 if branch == "candidate" else None,
                    success_steps=sum(successes),
                    terminal=successes[-1],
                    stable=all(successes[-40:]),
                    grasp_steps=grasp,
                    max_target_height_gain_m=max_height,
                    actual_command_changed=changed,
                )
            )
            env.close()
            env = None
        result.update(
            complete=True,
            passed=True,
            scope="Paired fixed-reference release-floor intervention; recorded arm commands, no responsive WAM or fresh qualification",
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
