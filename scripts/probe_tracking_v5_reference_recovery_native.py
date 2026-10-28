"""Bounded causal hold/resume arm pilot at actual post-hold state; CPU only."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import inverse, pose_error
from s2a.interior_joint_preview import build_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.quality_capped_inner_joint_plan import solve_interior_plan
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_reference_recovery_native_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    resume_ref = reference(REPO / "outputs/tracking_v5_reference_resume_numeric_20261028/verified.json")
    resume = read(resume_ref)
    require(resume["complete"] is resume["passed"] is True, "Closed actual post-hold numeric")
    hold = read(resume["hold_ref"])
    held = next(v for v in hold["branches"] if v["branch"] == "held")
    hold_reg = read(hold["registration_ref"])
    manifest = read(hold_reg["manifest_ref"])
    producer = read(hold_reg["producer_ref"])
    source_reg = read(producer["registration_ref"])
    worker = read(source_reg["worker_ref"])
    control_ref = reference(Path(worker["physical_case_dir"]) / "controller.json")
    control = read(control_ref)
    contract = control["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    prefix = np.asarray(read(held["commands_ref"]))
    require(prefix.shape == (378, 7), "Exactly378 actual prefix commands")
    with np.load(checked(held["states_ref"]), allow_pickle=False) as archive:
        prefix_states = archive["sim_states"].copy()
    profile = np.asarray(read(resume["history_update_ref"])["after_profile"])
    numeric = read(resume["hold_numeric_ref"])
    searches = read(next(v for v in numeric["branches"] if v["branch"] == "causal_error")["searches_ref"])
    old_plan = np.asarray(next(v for v in searches if v["step"] == 377)["held_quality"]["planned_commands"])
    warm = np.vstack((old_plan[1:], old_plan[-1]))
    chunk = read(source_reg["chunks_ref"])[46]
    targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
    tool = np.asarray(read(hold_reg["tool_ref"])["source_from_tcp"])
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ tool
    registration = publish(
        root / "registration.json",
        dict(
            source_ref=reference(Path(__file__)),
            resume_ref=resume_ref,
            hold_ref=resume["hold_ref"],
            manifest_ref=hold_reg["manifest_ref"],
            tool_ref=hold_reg["tool_ref"],
            control_ref=control_ref,
            chunks_ref=source_reg["chunks_ref"],
            seed=42,
            prefix_steps=378,
            maximum_continuation_steps=16,
            physics_budget=394,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            policy="Try next original cursor then hold last consumed cursor; same fixed shifted warm start,160 responses each; stop before dispatch if neither witness; stop after actual margin/quality violation",
            reference_phase="Start last-consumed cursor8,next9; resume advances exactly one; hold does not consume; no interpolation/deletion; original20Hz physical timing",
            gripper_policy="Retain last actually executed gripper command throughout: arm-only pilot, not a full coordinated retiming policy",
            causal_history="Nine original completed periods368..376 plus actual held377; update only own actual new25 inner states after dispatch; never borrow original378+",
            scope="Bounded native arm recovery pilot; not original timed tracking, complete path, WAM or fresh-task qualification",
        ),
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.current_coupled_state import read_current_dynamics
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import current_observation, make_environment, measured_twist, source_pose

    result = dict(complete=False, passed=False, registration_ref=registration, physics_steps=0)
    env = None
    states, commands, periods = [], list(prefix), []
    stop = None
    cursor = 8
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        robot = env.robots[0]
        controller = robot.controller
        robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
        limits = np.asarray(
            [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
        )
        previous = prefix[-1].copy()
        obs = current_observation(env)
        native_step = env.sim.step
        inner = []

        def observed_step(*args, **kwargs):
            native_step(*args, **kwargs)
            inner.append(
                dict(
                    time=float(env.sim.data.time),
                    q=np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes].copy(),
                    qvel=np.asarray(env.sim.data.qvel)[robot._ref_joint_vel_indexes].copy(),
                )
            )

        for step in range(395):
            sample, receipt = endpoint(env, step)
            state = env.sim.get_state().flatten().copy()
            states.append(state)
            if step <= 378:
                np.testing.assert_array_equal(state, prefix_states[step])
            if step in (0, 16, 56):
                render_anchor(env, step, sample, receipt, root)
            if step == 394:
                break
            if step % 8 == 0:
                render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
            if step < 378:
                env.step(prefix[step])
                result["physics_steps"] += 1
                obs = current_observation(env)
                continue
            site = np.eye(4)
            site[:3, :3] = env.sim.data.get_site_xmat(controller.eef_name)
            site[:3, 3] = env.sim.data.get_site_xpos(controller.eef_name)
            dynamics = read_current_dynamics(
                env,
                dict(
                    x_joint_position=[np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes]],
                    x_joint_velocity=[np.asarray(env.sim.data.qvel)[robot._ref_joint_vel_indexes]],
                    x_site_pose=[site],
                    x_site_twist=[measured_twist(env)],
                    x_previous_goal_ori=[np.asarray(controller.goal_ori).copy()],
                ),
                previous,
            )
            local = build_joint_preview(
                dynamics["mass"],
                dynamics["jacobian"],
                dynamics["kp"],
                dynamics["kd"],
                dynamics["robot_joint_position"],
                controller.initial_joint,
                null_kp=10.0,
                dt=0.002,
            )
            seed = np.tile(previous[:6], (8, 1))
            searches = []
            chosen = None
            for mode in ("resume", "hold"):
                indices = list(range(cursor + 1, cursor + 9)) if mode == "resume" else [cursor] * 8
                require(max(indices) < len(targets), "Only previously generated references")
                desired = targets[indices]
                twists = (
                    np.asarray([reference_world_twist(targets, k, anchor, 0.05)["twist"] for k in indices])
                    if mode == "resume"
                    else np.zeros((8, 6))
                )
                answer = solve_interior_plan(
                    local,
                    dynamics["robot_joint_velocity"],
                    seed,
                    previous[:6],
                    interface,
                    site,
                    site_from_tcp,
                    desired,
                    twists,
                    limits[:, 0],
                    limits[:, 1],
                    initial_guess=warm,
                    past_inner_error_rad=profile,
                )
                searches.append(dict(mode=mode, target_cursors=indices, answer=answer))
                if answer["witness_found"]:
                    chosen = searches[-1]
                    break
            if chosen is None:
                stop = dict(
                    step=step,
                    reason="no_model_witness",
                    action_dispatched=False,
                    searches=searches,
                    last_consumed_cursor=cursor,
                    history_before=profile.copy(),
                )
                break
            answer = chosen["answer"]
            command = np.r_[np.asarray(answer["planned_commands"])[0], prefix[-1, 6]]
            require(
                np.max(abs(command)) <= 1 + 1e-12 and np.max(abs(command[:6] - previous[:6])) <= 0.25 + 1e-12,
                "Original box and arm rate bounds",
            )
            require(answer["minimum_tightened_inner_slack_rad"] >= -1e-10, "All tightened inner points")
            predicted = np.asarray(answer["rollout"]["inner_joint_position"])[:25]
            before = profile.copy()
            inner.clear()
            env.sim.step = observed_step
            env.step(command)
            env.sim.step = native_step
            result["physics_steps"] += 1
            commands.append(command.copy())
            require(len(inner) == 25, "Exactly25 actual inner samples")
            np.testing.assert_allclose(np.diff([v["time"] for v in inner]), 0.002, atol=1e-12, rtol=0)
            actual = np.asarray([v["q"] for v in inner])
            residual = abs(actual - predicted)
            profile = np.maximum.accumulate(np.maximum(profile, residual), axis=0)
            if chosen["mode"] == "resume":
                cursor += 1
            obs = current_observation(env)
            error = pose_error(targets[cursor], source_pose(obs) @ tool)
            margin = float(np.minimum(actual - limits[:, 0], limits[:, 1] - actual).min())
            progress = object_progress(env)
            periods.append(
                dict(
                    step=step,
                    mode=chosen["mode"],
                    target_cursor=cursor,
                    command=command,
                    searches=searches,
                    actual_inner=list(inner),
                    history_before=before,
                    history_after=profile.copy(),
                    predicted_inner=predicted,
                    actual_tcp_error=error,
                    minimum_inner_margin_rad=margin,
                    success=bool(progress["success"]),
                    grasp=bool(progress["objects"]["akita_black_bowl_1"]["grasp_contact"]),
                )
            )
            previous = command.copy()
            plan = np.asarray(answer["planned_commands"])
            warm = np.vstack((plan[1:], plan[-1]))
            if (
                margin < 0.05 - 1e-12
                or np.linalg.norm(error[:3]) > 0.01 + 1e-10
                or np.linalg.norm(error[3:]) > 0.05 + 1e-10
            ):
                # Save actual final state before stopping; no more command is dispatched.
                states.append(env.sim.get_state().flatten().copy())
                stop = dict(
                    step=step + 1,
                    reason="actual_quality_or_margin_violation",
                    action_dispatched=False,
                    last_consumed_cursor=cursor,
                )
                break
        require(result["physics_steps"] == len(commands) == 378 + len(periods), "Exact dispatched period accounting")
        require(len(states) == len(commands) + 1, "Every actual endpoint including stop")
        path = root / "states.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, sim_states=np.asarray(states))
        result.update(
            complete=True,
            passed=True,
            states_ref=reference(path),
            commands_ref=publish(root / "commands.json", commands),
            periods_ref=publish(root / "periods.json", periods),
            stopped=stop,
            continuation_steps=len(periods),
            resumed_steps=sum(v["mode"] == "resume" for v in periods),
            held_steps=sum(v["mode"] == "hold" for v in periods),
            last_consumed_cursor=cursor,
            unused_cursors=list(range(cursor + 1, len(targets))),
            world_model_calls=0,
            fresh_cases=0,
            control_qualified=False,
            scope="Protocol completion only; bounded retimed arm pilot, not full coordinated trajectory or task qualification",
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
