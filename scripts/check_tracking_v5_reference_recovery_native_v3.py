"""Independent saved-action replay of bounded hold/resume pilot; no solver import."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import pose_error
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_reference_recovery_native_check_v3_20261028"
    require(not root.exists(), "No overwrite or retry")
    owner_ref = reference(REPO / "outputs/tracking_v5_reference_recovery_native_execution_20261028/result.json")
    owner = read(owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed actual producer owner")
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
    require(producer["complete"] is producer["passed"] is True, "Producer protocol complete")
    reg = read(producer["registration_ref"])
    manifest = read(reg["manifest_ref"])
    targets = np.asarray(read(reg["chunks_ref"])[46]["effective_targets_tcp"])
    tool = np.asarray(read(reg["tool_ref"])["source_from_tcp"])
    commands = np.asarray(read(producer["commands_ref"]))
    with np.load(checked(producer["states_ref"]), allow_pickle=False) as archive:
        saved_states = archive["sim_states"].copy()
    periods = read(producer["periods_ref"])
    require(len(commands) == 378 + len(periods) == producer["physics_steps"] <= 394, "Exact actual dispatch count")
    require(0 <= len(periods) <= 16 and len(saved_states) == len(commands) + 1, "All actual states")
    require(np.isfinite(commands).all() and np.max(abs(commands)) <= 1 + 1e-12, "Command box")
    require(np.max(abs(np.diff(np.vstack((np.zeros(7), commands)), axis=0)[:, :6])) <= 0.25 + 1e-12, "Arm rate")
    if len(periods):
        np.testing.assert_array_equal(commands[378:, 6], np.repeat(commands[377, 6], len(periods)))
    registration = publish(
        root / "registration.json",
        dict(
            producer_owner_ref=owner_ref,
            producer_ref=producer_ref,
            source_ref=reference(Path(__file__)),
            physics_budget=len(commands),
            numeric_absolute_tolerance=1e-10,
            numeric_relative_tolerance=0,
            sensor_refresh_protocol="One initial refresh, endpoint refresh before each action and one post-action refresh, matching producer; same render schedule",
            comparison_policy="No bitwise identity claim; state, inner q/qvel and reconstructed history within1e-10; commands and phase exact",
            world_model_calls=0,
            fresh_cases=0,
            seed=42,
            automatic_retry=False,
            scope="Independent every-state and continuation inner-state command replay with atol1e-10/rtol0; no planner imported; no full task qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import current_observation, make_environment, source_pose

    result = dict(complete=False, passed=False, registration_ref=registration, physics_steps=0)
    env = None
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        robot = env.robots[0]
        robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
        limits = np.asarray(
            [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
        )
        resume = read(reg["resume_ref"])
        profile = np.asarray(read(resume["history_update_ref"])["after_profile"])
        cursor = 8
        maximum_state_difference = 0.0
        maximum_inner_q_difference = 0.0
        maximum_inner_qvel_difference = 0.0
        inner, states, metrics = [], [], []
        # Match the producer sensor refresh protocol: current_observation invokes
        # sim.forward(), which can affect subsequent warm-started contact dynamics.
        obs = current_observation(env)
        native_step = env.sim.step

        def observed_step(*args, **kwargs):
            native_step(*args, **kwargs)
            inner.append(
                dict(
                    time=float(env.sim.data.time),
                    q=np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes].copy(),
                    qvel=np.asarray(env.sim.data.qvel)[robot._ref_joint_vel_indexes].copy(),
                )
            )

        for step in range(len(commands) + 1):
            sample, receipt = endpoint(env, step)
            state = env.sim.get_state().flatten().copy()
            states.append(state)
            maximum_state_difference = max(maximum_state_difference, float(np.max(abs(state - saved_states[step]))))
            np.testing.assert_allclose(state, saved_states[step], atol=1e-10, rtol=0)
            if step in (0, 16, 56):
                render_anchor(env, step, sample, receipt, root)
            if step == len(commands):
                break
            if step % 8 == 0:
                render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
            inner.clear()
            if step >= 378:
                env.sim.step = observed_step
            env.step(commands[step])
            env.sim.step = native_step
            result["physics_steps"] += 1
            obs = current_observation(env)
            if step < 378:
                continue
            row = periods[step - 378]
            require(row["step"] == step, "Consecutive actual feedback periods")
            require(row["mode"] in ("resume", "hold"), "Declared phase mode")
            searches = row["searches"]
            if row["mode"] == "resume":
                require(len(searches) == 1 and searches[0]["mode"] == "resume", "Advance first")
                np.testing.assert_array_equal(searches[0]["target_cursors"], np.arange(cursor + 1, cursor + 9))
                cursor += 1
            else:
                require(
                    len(searches) == 2
                    and searches[0]["mode"] == "resume"
                    and searches[0]["answer"]["witness_found"] is False
                    and searches[1]["mode"] == "hold",
                    "Hold only after resume no-witness",
                )
                np.testing.assert_array_equal(searches[1]["target_cursors"], np.repeat(cursor, 8))
            require(row["target_cursor"] == cursor, "No reference skipping or consumption on hold")
            require(searches[-1]["answer"]["witness_found"] is True, "Chosen recorded model witness")
            np.testing.assert_array_equal(commands[step], row["command"])
            np.testing.assert_array_equal(commands[step, :6], np.asarray(searches[-1]["answer"]["planned_commands"])[0])
            require(len(inner) == len(row["actual_inner"]) == 25, "Every actual2ms endpoint")
            q = np.asarray([v["q"] for v in inner])
            maximum_inner_q_difference = max(
                maximum_inner_q_difference, float(np.max(abs(q - np.asarray([v["q"] for v in row["actual_inner"]]))))
            )
            maximum_inner_qvel_difference = max(
                maximum_inner_qvel_difference,
                float(
                    np.max(
                        abs(
                            np.asarray([v["qvel"] for v in inner])
                            - np.asarray([v["qvel"] for v in row["actual_inner"]])
                        )
                    )
                ),
            )
            np.testing.assert_allclose(q, np.asarray([v["q"] for v in row["actual_inner"]]), atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                np.asarray([v["qvel"] for v in inner]),
                np.asarray([v["qvel"] for v in row["actual_inner"]]),
                atol=1e-10,
                rtol=0,
            )
            np.testing.assert_allclose(
                [v["time"] for v in inner], [v["time"] for v in row["actual_inner"]], atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(profile, row["history_before"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(profile, searches[-1]["answer"]["past_inner_error_rad"], atol=1e-10, rtol=0)
            profile = np.maximum.accumulate(np.maximum(profile, abs(q - np.asarray(row["predicted_inner"]))), axis=0)
            np.testing.assert_allclose(profile, row["history_after"], atol=1e-10, rtol=0)
            error = pose_error(targets[cursor], source_pose(obs) @ tool)
            margin = float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min())
            np.testing.assert_allclose(error, row["actual_tcp_error"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(margin, row["minimum_inner_margin_rad"], atol=1e-10, rtol=0)
            progress = object_progress(env)
            require(
                bool(progress["success"]) == row["success"]
                and bool(progress["objects"]["akita_black_bowl_1"]["grasp_contact"]) == row["grasp"],
                "Same task/grasp predicates",
            )
            good = (
                margin >= 0.05 - 1e-12
                and np.linalg.norm(error[:3]) <= 0.01 + 1e-10
                and np.linalg.norm(error[3:]) <= 0.05 + 1e-10
            )
            require(good or step == len(commands) - 1, "No action after actual guard violation")
            metrics.append(
                dict(
                    step=step,
                    target_cursor=cursor,
                    mode=row["mode"],
                    minimum_inner_margin_rad=margin,
                    actual_tcp_error=error,
                    actual_guard_satisfied=bool(good),
                    success=row["success"],
                    grasp=row["grasp"],
                )
            )
        require(cursor == producer["last_consumed_cursor"], "Exact final phase")
        np.testing.assert_array_equal(producer["unused_cursors"], np.arange(cursor + 1, len(targets)))
        if producer["stopped"] is not None:
            require(
                producer["stopped"]["step"] == len(commands) and producer["stopped"]["action_dispatched"] is False,
                "No command after explicit stop",
            )
        path = root / "states.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, sim_states=np.asarray(states))
        result.update(
            complete=True,
            passed=True,
            states_ref=reference(path),
            metrics_ref=publish(root / "metrics.json", metrics),
            maximum_state_difference=maximum_state_difference,
            maximum_inner_q_difference=maximum_inner_q_difference,
            maximum_inner_qvel_difference=maximum_inner_qvel_difference,
            continuation_steps=len(periods),
            last_consumed_cursor=cursor,
            world_model_calls=0,
            fresh_cases=0,
            scope="Independent replay within registered absolute1e-10 tolerance and exact phase/command audit only; no full coordinated timing or task qualification",
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
