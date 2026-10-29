"""Bounded forward-only FK-consistent native feedback pilot; CPU, no GPU model."""

import json
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.fk_pose_cap_scope_joint_plan import solve_interior_plan
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
    root = REPO / "outputs/tracking_v5_fk_first_feedback_native_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    scope_owner_ref = reference(REPO / "outputs/tracking_v5_fk_cap_scope_numeric_execution_20261028/result.json")
    owner = read(scope_owner_ref)
    require(owner["complete"] is owner["passed"] is True, "Closed matched-state scope comparison")
    exit_record, release = read(owner["owned_command"]["exit"]), read(owner["owned_command"]["release"])
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0,
        "Actual wait0/reap0",
    )
    scope_ref = owner["result_ref"]
    scope = read(scope_ref)
    sreg = read(scope["registration_ref"])
    selected = next(v for v in scope["entries"] if v["mode"] == "first_only")
    require(selected["answer"]["witness_found"] is True, "Verified same-state first-period witness")
    first = dict(answer=selected["answer"], current_q=scope["current_q"], current_velocity=scope["current_velocity"])
    prior_ref = sreg["native_ref"]
    prior = read(prior_ref)
    preg = read(prior["registration_ref"])
    with np.load(checked(prior["states_ref"]), allow_pickle=False) as archive:
        source_states = archive["sim_states"].copy()
    prefix = np.asarray(read(prior["commands_ref"]))
    require(prefix.shape == (370, 7) and len(source_states) == 371, "Exact certified actual prefix")
    manifest = read(preg["manifest_ref"])
    control = read(sreg["control_ref"])
    contract = control["contract"]
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    tool = np.asarray(read(sreg["tool_ref"])["source_from_tcp"])
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ tool
    chunk = read(preg["chunks_ref"])[46]
    targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
    fk = RobotSiteKinematics(
        checked(sreg["xml_ref"]).read_text(), contract["controller_site"], [f"robot0_joint{k}" for k in range(1, 8)]
    )
    registration = publish(
        root / "registration.json",
        dict(
            scope_ref=scope_ref,
            scope_owner_ref=scope_owner_ref,
            prior_native_ref=prior_ref,
            manifest_ref=preg["manifest_ref"],
            tool_ref=sreg["tool_ref"],
            chunks_ref=preg["chunks_ref"],
            source_ref=reference(Path(__file__)),
            seed=42,
            prefix_steps=370,
            maximum_continuation_steps=16,
            inherited_cursor_count=2,
            physics_budget=386,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            first_action="Archived actual370 matched-state first-only witness, exact state/current dynamics check; precomputed solve cost retained",
            subsequent_policy="FK first-period cap only, unchanged full8 pose/twist cost and all original joint/history/braking/terminal constraints; no witness stops before dispatch",
            history="Inherited only certified actual368/369; update using only own actually completed new periods",
            gripper_policy="Retain prefix last actual gripper command; arm-only, no coordinated task qualification",
            actual_guard="Every actual endpoint retains.01m/.05rad and every inner joint margin.05rad; stop next dispatch on violation",
            timing="Physical20Hz; solver wall-time retained separately, no realtime qualification",
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
    profile = np.asarray(scope["history_before"]).copy()
    np.testing.assert_array_equal(profile, first["answer"]["past_inner_error_rad"])
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

        for step in range(387):
            sample, receipt = endpoint(env, step)
            state = env.sim.get_state().flatten().copy()
            states.append(state)
            if step <= 370:
                np.testing.assert_array_equal(state, source_states[step])
            if step in (0, 16, 56):
                render_anchor(env, step, sample, receipt, root)
            if step == 386:
                break
            if step % 8 == 0:
                render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
            if step < 370:
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
            if step == 370:
                np.testing.assert_array_equal(dynamics["robot_joint_position"], first["current_q"])
                np.testing.assert_array_equal(dynamics["robot_joint_velocity"], first["current_velocity"])
                original_d = dict(
                    mass=scope["current_mass"], jacobian=scope["current_jacobian"], kp=scope["kp"], kd=scope["kd"]
                )
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
                    pose_cap_periods=1,
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
            len(states) == len(commands) + 1 and len(commands) == 370 + len(periods) == result["physics_steps"],
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
            last_consumed_cursor=len(periods) + 1,
            unused_cursors=list(range(len(periods) + 2, len(targets))),
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
