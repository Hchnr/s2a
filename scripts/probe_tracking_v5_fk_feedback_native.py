"""Bounded forward-only FK-consistent native feedback pilot; CPU, no GPU model."""

import json
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.fk_consistent_trajectory_joint_plan import solve_interior_plan
from s2a.generated_reference_twist import reference_world_twist
from s2a.geometry import inverse, pose_error
from s2a.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result as publish
from s2a.robot_site_kinematics import RobotSiteKinematics
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/tracking_v5_fk_feedback_native_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    check_owner_ref = reference(
        REPO / "outputs/tracking_v5_fk_consistent_trajectory_check_execution_20261028/result.json"
    )
    check_owner = read(check_owner_ref)
    require(check_owner["complete"] is check_owner["passed"] is True, "Closed independent MuJoCo geometry check")
    exit_record, release = read(check_owner["owned_command"]["exit"]), read(check_owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Actual wait0/reap0",
    )
    numeric_ref = reference(REPO / "outputs/tracking_v5_fk_consistent_trajectory_numeric_20261028/verified.json")
    numeric = read(numeric_ref)
    first = next(v for v in numeric["entries"] if v["horizon"] == 8)
    require(first["answer"]["witness_found"] is True, "All8 original-accuracy model witness")
    nreg = read(numeric["registration_ref"])
    producer = read(nreg["producer_ref"])
    reg = read(producer["registration_ref"])
    branch = next(v for v in producer["branches"] if v["branch"] == "causal_error")
    with np.load(checked(branch["states_ref"]), allow_pickle=False) as archive:
        source_states = archive["sim_states"].copy()
    originals = [json.loads(v) for v in checked(reg["rows_ref"]).read_text().splitlines()]
    prefix = np.asarray([v["command"] for v in originals[:368]])
    require(prefix.shape == (368, 7), "Exact original prefix")
    manifest = read(reg["manifest_ref"])
    control = read(nreg["control_ref"])
    contract = control["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    tool = np.asarray(read(nreg["tool_ref"])["source_from_tcp"])
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ tool
    chunk = read(reg["chunks_ref"])[46]
    targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
    fk = RobotSiteKinematics(
        checked(nreg["xml_ref"]).read_text(), contract["controller_site"], [f"robot0_joint{k}" for k in range(1, 8)]
    )
    registration = publish(
        root / "registration.json",
        dict(
            numeric_ref=numeric_ref,
            independent_geometry_owner_ref=check_owner_ref,
            original_producer_ref=nreg["producer_ref"],
            manifest_ref=reg["manifest_ref"],
            tool_ref=nreg["tool_ref"],
            chunks_ref=reg["chunks_ref"],
            source_ref=reference(Path(__file__)),
            seed=42,
            prefix_steps=368,
            maximum_continuation_steps=16,
            physics_budget=384,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            first_action="Archived same-state verified8-point plan; require exact state and matched current M/J/q/qvel before dispatch",
            subsequent_policy="Rebuild current robot model and solve FK-consistent8-point original reference preview; no-witness stops before dispatch; no hold/skip/relaxation",
            history="Empty at368; only this branch's completed25-sample residuals update own monotone inner profile",
            gripper_policy="Keep original prefix last gripper command; arm-only pilot, no full coordinated task qualification",
            actual_guard="Stop next dispatch on actual TCP>.01m/.05rad or actual inner margin<.05rad",
            timing="Physical20Hz/25x2ms; optimizer wall-time recorded separately, not realtime qualification",
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
    stopped = None
    profile = np.zeros((25, 7))
    previous = prefix[-1].copy()
    warm = np.asarray(first["answer"]["planned_commands"])
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        robot, controller = env.robots[0], env.robots[0].controller
        robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
        limits = np.asarray(
            [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
        )
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

        for step in range(385):
            sample, receipt = endpoint(env, step)
            state = env.sim.get_state().flatten().copy()
            states.append(state)
            if step <= 368:
                np.testing.assert_array_equal(state, source_states[step])
            if step in (0, 16, 56):
                render_anchor(env, step, sample, receipt, root)
            if step == 384:
                break
            if step % 8 == 0:
                render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
            if step < 368:
                env.step(prefix[step])
                result["physics_steps"] += 1
                obs = current_observation(env)
                continue
            site = np.eye(4)
            site[:3, :3] = env.sim.data.get_site_xmat(controller.eef_name)
            site[:3, 3] = env.sim.data.get_site_xpos(controller.eef_name)
            current = dict(
                x_joint_position=[np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes]],
                x_joint_velocity=[np.asarray(env.sim.data.qvel)[robot._ref_joint_vel_indexes]],
                x_site_pose=[site],
                x_site_twist=[measured_twist(env)],
                x_previous_goal_ori=[np.asarray(controller.goal_ori).copy()],
            )
            dynamics = read_current_dynamics(env, current, previous)
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
            cursor = step - 368
            desired = targets[cursor : cursor + 8]
            twists = np.asarray(
                [reference_world_twist(targets, k, anchor, 0.05)["twist"] for k in range(cursor, cursor + 8)]
            )
            started = time.monotonic()
            if step == 368:
                np.testing.assert_array_equal(dynamics["robot_joint_position"], first["current_q"])
                np.testing.assert_array_equal(dynamics["robot_joint_velocity"], first["current_velocity"])
                original_d = read(branch["steps_ref"])[0]["dynamics"]
                for name in ("mass", "jacobian", "kp", "kd"):
                    np.testing.assert_allclose(dynamics[name], original_d[name], atol=1e-10, rtol=0)
                answer = first["answer"]
                route = "archived_same_state_plan"
            else:
                answer = solve_interior_plan(
                    local,
                    dynamics["robot_joint_velocity"],
                    np.tile(previous[:6], (8, 1)),
                    previous[:6],
                    interface,
                    site,
                    site_from_tcp,
                    desired,
                    twists,
                    limits[:, 0],
                    limits[:, 1],
                    fk_jacobian=fk,
                    initial_guess=warm,
                    past_inner_error_rad=profile,
                )
                route = "current_feedback_search"
            compute_seconds = time.monotonic() - started
            if not answer["witness_found"]:
                stopped = dict(
                    step=step,
                    reason="no_model_witness",
                    action_dispatched=False,
                    answer=answer,
                    history_before=profile.copy(),
                )
                break
            command = np.r_[np.asarray(answer["planned_commands"])[0], prefix[-1, 6]]
            require(
                np.max(abs(command)) <= 1 + 1e-12 and np.max(abs(command[:6] - previous[:6])) <= 0.25 + 1e-12,
                "Original box/rate",
            )
            prediction = rollout_joint_preview(
                local,
                dynamics["robot_joint_velocity"],
                np.asarray([interface.decode(command[:6])]),
                site,
                limits[:, 0],
                limits[:, 1],
                enforce_limits=False,
            )
            predicted = prediction["inner_joint_position"]
            np.testing.assert_allclose(
                predicted, np.asarray(answer["rollout"]["inner_joint_position"])[:25], atol=1e-10, rtol=0
            )
            before = profile.copy()
            inner.clear()
            env.sim.step = observed_step
            env.step(command)
            env.sim.step = native_step
            result["physics_steps"] += 1
            commands.append(command.copy())
            require(len(inner) == 25, "Every actual2ms inner sample")
            actual = np.asarray([v["q"] for v in inner])
            profile = np.maximum.accumulate(np.maximum(profile, abs(actual - predicted)), axis=0)
            obs = current_observation(env)
            error = pose_error(targets[cursor], source_pose(obs) @ tool)
            margin = float(np.minimum(actual - limits[:, 0], limits[:, 1] - actual).min())
            progress = object_progress(env)
            periods.append(
                dict(
                    step=step,
                    cursor=cursor,
                    route=route,
                    command=command,
                    answer=answer,
                    current_dynamics=dynamics,
                    compute_seconds=compute_seconds,
                    history_before=before,
                    history_after=profile.copy(),
                    predicted_inner=predicted,
                    actual_inner=list(inner),
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
                states.append(env.sim.get_state().flatten().copy())
                stopped = dict(step=step + 1, reason="actual_quality_or_margin_violation", action_dispatched=False)
                break
        require(
            len(states) == len(commands) + 1 and len(commands) == 368 + len(periods) == result["physics_steps"],
            "Every dispatched action/endpoint",
        )
        path = root / "states.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, sim_states=np.asarray(states))
        result.update(
            complete=True,
            passed=True,
            states_ref=reference(path),
            commands_ref=publish(root / "commands.json", commands),
            periods_ref=publish(root / "periods.json", periods),
            stopped=stopped,
            continuation_steps=len(periods),
            last_consumed_cursor=len(periods) - 1,
            unused_cursors=list(range(len(periods), len(targets))),
            world_model_calls=0,
            fresh_cases=0,
            control_qualified=False,
            scope="Bounded actual forward arm feedback; not full coordinated trajectory, WAM/fresh task or realtime qualification",
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
