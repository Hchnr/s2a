"""Independently replay one first-period FK preview horizon; no solver import."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import inverse, pose_error
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF
from s2a.xml_site_kinematics import XMLSiteKinematics


def main(horizon):
    strict_native_environment()
    root = REPO / f"outputs/tracking_v5_fk_first_long_h{horizon}_native_check_20261028"
    require(not root.exists(), "No overwrite or retry")
    owner_ref = reference(REPO / f"outputs/tracking_v5_fk_first_long_h{horizon}_native_execution_20261028/result.json")
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
    require(reg["horizon"] == horizon, "Exact registered preview horizon")
    require(reg["prefix_steps"] == 368 and reg["maximum_continuation_steps"] == 16, "Original common prefix and budget")
    require(
        np.isfinite(reg["archived_first_search_seconds"]) and reg["archived_first_search_seconds"] >= 0,
        "Archived first-plan compute cost retained",
    )
    numeric = read(reg["numeric_ref"])
    require(numeric["complete"] is numeric["passed"] is True, "Closed numeric source")
    numeric_registration = read(numeric["registration_ref"])
    numeric_entry = next(v for v in numeric["entries"] if v["horizon"] == horizon)
    np.testing.assert_array_equal(reg["archived_first_search_seconds"], numeric_entry["compute_seconds"])
    control_ref, xml_ref = numeric_registration["control_ref"], numeric_registration["xml_ref"]
    control = read(control_ref)
    kinematics = XMLSiteKinematics(checked(xml_ref).read_text(), control["contract"]["controller_site"])
    robot_nodes = {node.get("name"): indices for node, indices in kinematics.indices.items()}
    robot_indices = [robot_nodes[f"robot0_joint{k}"] for k in range(1, 8)]
    require(all(indices[2] in ("hinge", "slide") for indices in robot_indices), "Scalar robot joint coordinates")
    qids = np.asarray([indices[0] for indices in robot_indices])
    dofs = np.asarray([indices[1] for indices in robot_indices])
    manifest = read(reg["manifest_ref"])
    targets = np.asarray(read(reg["chunks_ref"])[46]["effective_targets_tcp"])
    tool = np.asarray(read(reg["tool_ref"])["source_from_tcp"])
    site_from_tcp = inverse(np.asarray(control["source_from_site"])) @ tool
    commands = np.asarray(read(producer["commands_ref"]))
    with np.load(checked(producer["states_ref"]), allow_pickle=False) as archive:
        saved_states = archive["sim_states"].copy()
    periods = read(producer["periods_ref"])
    require(producer["continuation_steps"] == len(periods), "Actual continuation count")
    require(commands.shape == (368 + len(periods), 7), "Every dispatched command has seven entries")
    require(len(commands) == 368 + len(periods) == producer["physics_steps"] <= 384, "Exact actual dispatch count")
    require(0 <= len(periods) <= 16 and len(saved_states) == len(commands) + 1, "All actual states")
    require(np.isfinite(commands).all() and np.max(abs(commands)) <= 1 + 1e-12, "Command box")
    require(np.max(abs(np.diff(np.vstack((np.zeros(7), commands)), axis=0)[:, :6])) <= 0.25 + 1e-12, "Arm rate")
    if len(periods):
        np.testing.assert_array_equal(commands[368:, 6], np.repeat(commands[367, 6], len(periods)))
    registration = publish(
        root / "registration.json",
        dict(
            horizon=horizon,
            producer_owner_ref=owner_ref,
            producer_ref=producer_ref,
            numeric_ref=reg["numeric_ref"],
            control_ref=control_ref,
            xml_ref=xml_ref,
            tool_ref=reg["tool_ref"],
            source_ref=reference(Path(__file__)),
            physics_budget=len(commands),
            numeric_absolute_tolerance=1e-10,
            numeric_relative_tolerance=0,
            sensor_refresh_protocol="One initial refresh, endpoint refresh before each action and one post-action refresh, matching producer; same render schedule",
            comparison_policy="No bitwise identity claim; state, inner q/qvel and reconstructed history within1e-10; commands and phase exact",
            prediction_geometry_policy="Independent XML FK/twist and TCP errors from copies of each actual pre-action state with saved predicted robot q/qvel; no simulation mutation",
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
        profile = np.zeros((25, 7))
        cursor = -1
        maximum_state_difference = 0.0
        maximum_inner_q_difference = 0.0
        maximum_inner_qvel_difference = 0.0
        maximum_predicted_pose_difference = 0.0
        maximum_predicted_twist_difference = 0.0
        maximum_predicted_error_difference = 0.0
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
            if step >= 368:
                env.sim.step = observed_step
            env.step(commands[step])
            env.sim.step = native_step
            result["physics_steps"] += 1
            obs = current_observation(env)
            if step < 368:
                continue
            row = periods[step - 368]
            require(row["step"] == step, "Consecutive actual feedback periods")
            cursor += 1
            require(row["cursor"] == cursor == step - 368, "Forward-only reference, no skip or hold")
            answer = row["answer"]
            require(answer["witness_found"] is True, "Chosen original-accuracy model witness")
            require(
                answer["pose_caps_horizon"] == 1 and answer["all_pose_caps_enforced"] is False,
                "Declared first-period pose caps and full future error visibility",
            )
            plan = np.asarray(answer["planned_commands"])
            saved_predicted_errors = np.asarray(answer["all_tcp_errors"])
            require(plan.shape == (horizon, 6) and np.isfinite(plan).all(), "Exact finite preview command horizon")
            require(
                saved_predicted_errors.shape == (horizon, 6) and np.isfinite(saved_predicted_errors).all(),
                "Every future predicted error remains visible",
            )
            rollout = answer["rollout"]
            predicted_q, predicted_velocity = (
                np.asarray(rollout["joint_position"]),
                np.asarray(rollout["joint_velocity"]),
            )
            saved_predicted_pose, saved_predicted_twist = np.asarray(rollout["pose"]), np.asarray(rollout["twist"])
            require(
                predicted_q.shape == predicted_velocity.shape == (horizon, 7)
                and saved_predicted_pose.shape == (horizon, 4, 4)
                and saved_predicted_twist.shape == (horizon, 6),
                "Complete predicted joint, pose and twist horizon",
            )
            predicted_errors = []
            for period in range(horizon):
                changed = saved_states[step].copy()
                changed[1 + qids] = predicted_q[period]
                changed[1 + kinematics.nq + dofs] = predicted_velocity[period]
                xml_pose, xml_twist = kinematics.endpoint(changed)
                xml_error = pose_error(targets[cursor + period], xml_pose @ site_from_tcp)
                maximum_predicted_pose_difference = max(
                    maximum_predicted_pose_difference, float(np.max(abs(xml_pose - saved_predicted_pose[period])))
                )
                maximum_predicted_twist_difference = max(
                    maximum_predicted_twist_difference, float(np.max(abs(xml_twist - saved_predicted_twist[period])))
                )
                maximum_predicted_error_difference = max(
                    maximum_predicted_error_difference, float(np.max(abs(xml_error - saved_predicted_errors[period])))
                )
                np.testing.assert_allclose(xml_pose, saved_predicted_pose[period], atol=1e-10, rtol=0)
                np.testing.assert_allclose(xml_twist, saved_predicted_twist[period], atol=1e-10, rtol=0)
                np.testing.assert_allclose(xml_error, saved_predicted_errors[period], atol=1e-10, rtol=0)
                predicted_errors.append(xml_error)
            predicted_errors = np.asarray(predicted_errors)
            np.testing.assert_allclose(answer["first_tcp_error"], predicted_errors[0], atol=1e-10, rtol=0)
            require(
                np.linalg.norm(predicted_errors[0, :3]) <= 0.01 + 1e-10
                and np.linalg.norm(predicted_errors[0, 3:]) <= 0.05 + 1e-10,
                "First predicted endpoint keeps original pose accuracy",
            )
            require(np.isfinite(row["compute_seconds"]) and row["compute_seconds"] >= 0, "Actual planning duration")
            np.testing.assert_array_equal(commands[step], row["command"])
            np.testing.assert_array_equal(commands[step, :6], np.asarray(answer["planned_commands"])[0])
            np.testing.assert_allclose(
                row["predicted_inner"], np.asarray(answer["rollout"]["inner_joint_position"])[:25], atol=1e-10, rtol=0
            )
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
            np.testing.assert_allclose(profile, answer["past_inner_error_rad"], atol=1e-10, rtol=0)
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
                    route=row["route"],
                    minimum_inner_margin_rad=margin,
                    actual_tcp_error=error,
                    actual_guard_satisfied=bool(good),
                    predicted_tcp_errors=predicted_errors,
                    predicted_endpoint_caps_satisfied=(
                        (np.linalg.norm(predicted_errors[:, :3], axis=1) <= 0.01 + 1e-10)
                        & (np.linalg.norm(predicted_errors[:, 3:], axis=1) <= 0.05 + 1e-10)
                    ).tolist(),
                    success=row["success"],
                    grasp=row["grasp"],
                )
            )
        require(cursor == producer["last_consumed_cursor"], "Exact final phase")
        np.testing.assert_array_equal(producer["unused_cursors"], np.arange(cursor + 1, len(targets)))
        stop = producer["stopped"]
        all_actual_guards_satisfied = all(v["actual_guard_satisfied"] for v in metrics)
        if stop is not None:
            require(
                stop["step"] == len(commands) and stop["action_dispatched"] is False,
                "No command after explicit stop",
            )
            if stop["reason"] == "no_model_witness":
                require(stop["answer"]["witness_found"] is False, "No model witness at the stopped endpoint")
                require(
                    stop["answer"]["pose_caps_horizon"] == 1 and stop["answer"]["all_pose_caps_enforced"] is False,
                    "Stopped search used the same first-period pose policy",
                )
                require(np.isfinite(stop["compute_seconds"]) and stop["compute_seconds"] >= 0, "Stop search duration")
                np.testing.assert_allclose(profile, stop["history_before"], atol=1e-10, rtol=0)
                np.testing.assert_allclose(profile, stop["answer"]["past_inner_error_rad"], atol=1e-10, rtol=0)
                require(all_actual_guards_satisfied, "No attempted search after actual guard violation")
            else:
                require(stop["reason"] == "actual_quality_or_margin_violation", "Known predeclared stop reason")
                require(bool(metrics) and not metrics[-1]["actual_guard_satisfied"], "Actual violation explains stop")
        else:
            require(len(periods) == 16 and all_actual_guards_satisfied, "Unstopped branch exhausted its fixed budget")
        path = root / "states.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, sim_states=np.asarray(states))
        result.update(
            complete=True,
            passed=True,
            horizon=horizon,
            all_actual_guards_satisfied=all_actual_guards_satisfied,
            states_ref=reference(path),
            metrics_ref=publish(root / "metrics.json", metrics),
            maximum_state_difference=maximum_state_difference,
            maximum_inner_q_difference=maximum_inner_q_difference,
            maximum_inner_qvel_difference=maximum_inner_qvel_difference,
            maximum_predicted_pose_difference=maximum_predicted_pose_difference,
            maximum_predicted_twist_difference=maximum_predicted_twist_difference,
            maximum_predicted_error_difference=maximum_predicted_error_difference,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=(8, 12, 16), required=True)
    args = parser.parse_args()
    main(args.horizon)
