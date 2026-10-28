"""Paired one-period explicit reference hold; not original timed qualification."""

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
    root = REPO / "outputs/tracking_v5_reference_hold_native_20261028"
    require(not root.exists(), "No overwrite or retry")
    numeric_ref = reference(REPO / "outputs/tracking_v5_reference_hold_numeric_20261028/verified.json")
    numeric = read(numeric_ref)
    require(numeric["complete"] is numeric["passed"] is True, "Closed hold numeric")
    numeric_branch = next(v for v in numeric["branches"] if v["branch"] == "causal_error")
    entry = next(v for v in read(numeric_branch["searches_ref"]) if v["step"] == 377)
    require(entry["held_quality"]["witness_found"] is True, "Actual held-reference witness")
    producer_ref = reference(REPO / "outputs/tracking_v5_causal_inner_continuation_20261028/result.json")
    producer = read(producer_ref)
    reg = read(producer["registration_ref"])
    source_branch = next(v for v in producer["branches"] if v["branch"] == "causal_error")
    rows = read(source_branch["steps_ref"])
    originals = [json.loads(v) for v in checked(reg["rows_ref"]).read_text().splitlines()]
    with np.load(checked(source_branch["states_ref"]), allow_pickle=False) as archive:
        source_states = archive["sim_states"].copy()
    manifest = read(reg["manifest_ref"])
    tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
    tool = np.asarray(read(tool_ref)["source_from_tcp"])
    targets = np.asarray(read(reg["chunks_ref"])[46]["effective_targets_tcp"])
    require(entry["held_cursor"] == 8 and entry["original_cursor"] == 9, "Exact explicit phase change")
    np.testing.assert_array_equal(entry["held_target_tcp"], targets[8])
    prefix = np.asarray([r["command"] for r in originals[:368]] + [r["command"] for r in rows[:9]])
    require(prefix.shape == (377, 7), "Exact actually executed prefix")
    timed = np.asarray(rows[9]["command"])
    held = timed.copy()
    held[:6] = np.asarray(entry["held_quality"]["planned_commands"])[0]
    registration = publish(
        root / "registration.json",
        dict(
            numeric_ref=numeric_ref,
            producer_ref=producer_ref,
            source_ref=reference(Path(__file__)),
            source_states_ref=source_branch["states_ref"],
            source_steps_ref=source_branch["steps_ref"],
            manifest_ref=reg["manifest_ref"],
            tool_ref=tool_ref,
            intervention_step=377,
            timed_target=targets[9],
            held_target=targets[8],
            gripper_policy="Same saved gripper command in both branches; arm-reference intervention only",
            physics_budget=756,
            world_model_calls=0,
            fresh_cases=0,
            seed=42,
            automatic_retry=False,
            scope="One explicit held-reference arm action; no original-timing or complete retimed-path qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.geometry import pose_error
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import current_observation, make_environment, source_pose

    result = dict(complete=False, passed=False, registration_ref=registration, physics_steps=0, branches=[])
    env = None
    try:
        for branch, command in (("timed", timed), ("held", held)):
            commands = np.vstack((prefix, command))
            require(np.isfinite(commands).all() and np.max(abs(commands)) <= 1 + 1e-12, "Native command box")
            require(
                np.max(abs(np.diff(np.vstack((np.zeros(7), commands)), axis=0)[:, :6])) <= 0.25 + 1e-12, "Original rate"
            )
            env, _ = make_environment(manifest["runtime_config"], root / branch / "native", manifest["record"], 0)
            robot = env.robots[0]
            obs = current_observation(env)
            robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            native_step = env.sim.step
            inner, states = [], []

            def observed_step(*args, **kwargs):
                native_step(*args, **kwargs)
                inner.append(
                    dict(
                        time=float(env.sim.data.time),
                        q=np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes].copy(),
                        qvel=np.asarray(env.sim.data.qvel)[robot._ref_joint_vel_indexes].copy(),
                    )
                )

            for i in range(379):
                sample, receipt = endpoint(env, i)
                state = env.sim.get_state().flatten().copy()
                states.append(state)
                if i <= 377 or branch == "timed":
                    np.testing.assert_array_equal(state, source_states[i])
                if i in (0, 16, 56):
                    render_anchor(env, i, sample, receipt, root / branch)
                if i == 378:
                    break
                if i % 8 == 0:
                    render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
                if i == 377:
                    env.sim.step = observed_step
                env.step(commands[i])
                env.sim.step = native_step
                result["physics_steps"] += 1
                obs = current_observation(env)
            require(len(inner) == 25, "All actual intervention inner steps")
            q = np.asarray([v["q"] for v in inner])
            path = root / branch / "states.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, sim_states=np.asarray(states))
            progress = object_progress(env)
            actual_tcp = source_pose(obs) @ tool
            result["branches"].append(
                dict(
                    branch=branch,
                    states_ref=reference(path),
                    commands_ref=publish(root / branch / "commands.json", commands),
                    inner_ref=publish(root / branch / "inner.json", inner),
                    minimum_inner_margin_rad=float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min()),
                    actual_tcp=actual_tcp,
                    held_reference_error=pose_error(targets[8], actual_tcp),
                    original_timed_reference_error=pose_error(targets[9], actual_tcp),
                    success=bool(progress["success"]),
                    grasp=bool(progress["objects"]["akita_black_bowl_1"]["grasp_contact"]),
                )
            )
            env.close()
            env = None
        require(result["physics_steps"] == 756, "Exact two377+1 branches")
        result.update(
            complete=True,
            passed=True,
            world_model_calls=0,
            fresh_cases=0,
            scope="One-period explicit timing intervention only; no full tracking/task qualification",
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
