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
    root = REPO / "outputs/tracking_v5_reference_hold_native_check_20261028"
    require(not root.exists(), "No overwrite or retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_reference_hold_native_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed producer owner")
    exit_record, release = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Actual wait0 and reap0",
    )
    producer_ref = owner["result_ref"]
    producer = read(producer_ref)
    require(
        producer["complete"] is producer["passed"] is True and producer["physics_steps"] == 756,
        "Actual complete two branches",
    )
    reg = read(producer["registration_ref"])
    manifest = read(reg["manifest_ref"])
    tool = np.asarray(read(reg["tool_ref"])["source_from_tcp"])
    held_target, timed_target = np.asarray(reg["held_target"]), np.asarray(reg["timed_target"])
    require([v["branch"] for v in producer["branches"]] == ["timed", "held"], "Both declared branches")
    registration = publish(
        root / "registration.json",
        dict(
            producer_owner_ref=owner_ref,
            producer_ref=producer_ref,
            source_ref=reference(Path(__file__)),
            physics_budget=756,
            world_model_calls=0,
            fresh_cases=0,
            seed=42,
            automatic_retry=False,
            scope="Independent saved-command physical replay only; no solver or timing-governor import",
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
        for source_branch in producer["branches"]:
            branch = source_branch["branch"]
            commands = np.asarray(read(source_branch["commands_ref"]))
            require(commands.shape == (378, 7), "Exact actual command count")
            with np.load(checked(source_branch["states_ref"]), allow_pickle=False) as archive:
                source_states = archive["sim_states"].copy()
            saved_inner = read(source_branch["inner_ref"])
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
            np.testing.assert_array_equal(q, np.asarray([v["q"] for v in saved_inner]))
            np.testing.assert_array_equal(
                np.asarray([v["qvel"] for v in inner]), np.asarray([v["qvel"] for v in saved_inner])
            )
            np.testing.assert_allclose([v["time"] for v in inner], [v["time"] for v in saved_inner], atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min()),
                source_branch["minimum_inner_margin_rad"],
                atol=1e-12,
                rtol=0,
            )
            path = root / branch / "states.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, sim_states=np.asarray(states))
            progress = object_progress(env)
            actual_tcp = source_pose(obs) @ tool
            np.testing.assert_allclose(
                pose_error(held_target, actual_tcp), source_branch["held_reference_error"], atol=1e-12, rtol=0
            )
            np.testing.assert_allclose(
                pose_error(timed_target, actual_tcp),
                source_branch["original_timed_reference_error"],
                atol=1e-12,
                rtol=0,
            )
            require(bool(progress["success"]) == source_branch["success"], "Same task predicate")
            require(
                bool(progress["objects"]["akita_black_bowl_1"]["grasp_contact"]) == source_branch["grasp"],
                "Same grasp predicate",
            )
            result["branches"].append(
                dict(
                    branch=branch,
                    states_ref=reference(path),
                    commands_ref=publish(root / branch / "commands.json", commands),
                    inner_ref=publish(root / branch / "inner.json", inner),
                    minimum_inner_margin_rad=float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min()),
                    actual_tcp=actual_tcp,
                    held_reference_error=pose_error(held_target, actual_tcp),
                    original_timed_reference_error=pose_error(timed_target, actual_tcp),
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
            scope="Independent all-state/inner/velocity saved-command replay of timing intervention; no full tracking or task qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        publish(root / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
