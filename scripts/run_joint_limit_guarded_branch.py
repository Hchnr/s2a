"""Fixed saved-reference joint-limit candidate with original prefix and native goals."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, write_json
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/joint_limit_guarded_branches_20261027"
PREPARED = REPO / "outputs/preview_chunk_diagnostic_preparation_20261027/prepared.json"


def execute(index, horizon):
    strict_native_environment()
    prepared_ref = reference(PREPARED)
    require(
        prepared_ref["sha256"] == "79d07e233349af103ba935f9d968cc1de0ef7049e7aa2bbdfe9bc9869da30d80",
        "Fixed diagnostic preparation required",
    )
    prepared = read(prepared_ref)
    case = prepared["cases"][index]
    output = ROOT / f"case{index:02d}_h{horizon}"
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        case=case["case"],
        horizon=horizon,
        prepared_ref=prepared_ref,
        source_ref=reference(Path(__file__)),
        prefix_steps=0,
        branch_steps=0,
        new_model_calls=0,
        fresh_cases=0,
    )
    env = None
    records = []
    try:
        old_prepared = read(reference(REPO / "outputs/serialized_preview_wam_development_20261026/prepared.json"))
        for path, digest in old_prepared["source_sha256"].items():
            checked(dict(path=path, sha256=digest))
        for ref in case["original_refs"]:
            checked(ref)
        guard = read(CONTROL_REF)["cv2_guard_registration"]
        restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
        from collect_wam_native_training import endpoint, render_anchor

        from s2a.cameras import render_current_cameras
        from s2a.checkpoint_simulation import object_progress
        from s2a.coupled_pose_preview_runtime import solve_preview
        from s2a.coupled_response_history import CoupledResponseHistory
        from s2a.current_coupled_response import build_response
        from s2a.current_coupled_state import read_current_dynamics
        from s2a.current_model_state_native_precision import current_model_state
        from s2a.geometry import pose_error
        from s2a.gripper_motor_state import preserve_motor_target
        from s2a.joint_limit_guarded_native import refine_native_guarded as refine_native
        from s2a.joint_limit_native import JointLimitResponseHistory
        from s2a.libero_simulation import aperture, current_observation, make_environment, measured_twist, source_pose
        from s2a.loaded_velocity_control import LoadedVelocityControl
        from s2a.runtime_response_history import RuntimeResponseHistory

        refs = {Path(r["path"]).name: r for r in case["original_refs"]}
        worker, chunks = read(refs["worker_result.json"]), read(refs["chunks.json"])
        rows = [json.loads(line) for line in Path(refs["steps.jsonl"]["path"]).read_text().splitlines()]
        receipts = [json.loads(line) for line in checked(case["endpoint_receipts_ref"]).read_text().splitlines()]
        config, manifest = read(case["config_ref"]), read(case["manifest_ref"])
        checked(worker["arrays_ref"])
        contract_record = read(reference(Path(refs["chunks.json"]["path"]).parent / "controller.json"))
        contract, mapping = contract_record["contract"], np.asarray(contract_record["source_from_site"])
        h = np.asarray(read(case["tool_ref"])["source_from_tcp"])
        env, _ = make_environment(manifest["runtime_config"], output, manifest["record"], 0)
        obs = current_observation(env)
        env.robots[0].gripper.current_action = preserve_motor_target(env.robots[0].gripper.current_action)
        grip = LoadedVelocityControl(
            read(case["parameter_ref"]), aperture(obs), velocity_envelope=config["gripper_velocity_envelope"]
        )
        prefix = case["replay_periods"]
        previous_grip = 0.0
        for step in range(prefix + 1):
            sample, receipt = endpoint(env, step)
            if step in (0, 16, 56):
                render_anchor(env, step, sample, receipt, output)
            require(receipt["sim_state_sha256"] == receipts[step]["sim_state_sha256"], f"Prefix state differs at{step}")
            if step == prefix:
                break
            row, chunk = rows[step], chunks[step // 8]
            update = grip.update(
                row["opening_before"],
                np.asarray(row["motor_target_before"]),
                row["measured_period_velocity"],
                np.asarray(chunk["effective_targets_tcp"]),
                np.asarray(chunk["openings"]),
                step % 8,
                np.asarray(row["measured_source"]) @ h,
                previous_command=previous_grip,
                dt=0.05,
            )
            require(abs(update.command - row["command"][-1]) <= 1e-12, "Prefix gripper differs")
            previous_grip = update.command
            if step % 8 == 0:
                render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
            env.step(np.asarray(row["command"]))
            obs = current_observation(env)
            result["prefix_steps"] += 1
        chunk = chunks[case["event"]["query_index"]]
        render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
        generated = np.asarray(chunk["effective_targets_tcp"])
        generated_site = generated @ np.linalg.inv(h) @ mapping
        anchor_site = np.asarray(chunk["anchor_tcp"]) @ np.linalg.inv(h) @ mapping
        history = CoupledResponseHistory()
        limited_history = JointLimitResponseHistory()
        history.innovation = np.asarray(rows[prefix - 1]["coupled_response_history"]["innovation_after"]).copy()
        history.completed_periods = prefix
        previous = np.asarray(rows[prefix - 1]["command"]).copy()
        result["branch_initial_state_sha256"] = receipt["sim_state_sha256"]
        result["branch_initial_motor"] = env.robots[0].gripper.current_action.copy()
        for j in range(horizon):
            measured = source_pose(obs)
            velocity = np.asarray(obs["robot0_gripper_qvel"])
            update = grip.update(
                aperture(obs),
                env.robots[0].gripper.current_action,
                0.05 * (velocity[0] - velocity[1]) / 0.079,
                generated,
                np.asarray(chunk["openings"]),
                j,
                measured @ h,
                previous_command=previous_grip,
                dt=0.05,
            )
            # The original SOURCE inverse is provisional only: the preview
            # overwrites all six arm commands. Its residual is not a preview input.
            current, _ = current_model_state(
                env, previous, previous, previous_available=True, history=RuntimeResponseHistory()
            )
            initial = read_current_dynamics(env, current, previous)
            site, twist = current["x_site_pose"][0], current["x_site_twist"][0]
            solved = solve_preview(
                initial["mass"],
                initial["jacobian"],
                site,
                twist,
                generated_site,
                anchor_site,
                j,
                previous[:6],
                contract,
                innovation=history.current(site, twist),
            )
            refined = refine_native(
                env,
                initial,
                site,
                twist,
                generated_site,
                anchor_site,
                j,
                solved["command"],
                previous[:6],
                contract,
                limited_history,
                site_from_tcp=np.linalg.inv(mapping) @ h,
            )
            command = np.r_[refined["command"], update.command]
            limited_history.commit_prediction(site, twist, refined["nominal_response"])
            dynamics = read_current_dynamics(env, current, command)
            model = build_response(
                dynamics["mass"],
                dynamics["jacobian"],
                dynamics["kp"],
                dynamics["kd"],
                dynamics["relative_rotation_goal"],
            )
            history.commit_executed(site, twist, model, command[:3], contract)
            env.step(command)
            obs = current_observation(env)
            np.testing.assert_allclose(
                env.robots[0].controller.goal_pos, refined["expected_goal_pos"], atol=1e-9, rtol=0
            )
            np.testing.assert_allclose(
                env.robots[0].controller.goal_ori, refined["expected_goal_ori"], atol=1e-8, rtol=0
            )
            site_after = np.eye(4)
            site_after[:3, :3] = env.sim.data.get_site_xmat(env.robots[0].controller.eef_name)
            site_after[:3, 3] = env.sim.data.get_site_xpos(env.robots[0].controller.eef_name)
            history.observe_completed(site_after, measured_twist(env))
            limited_history.observe_completed(site_after, measured_twist(env))
            sample, receipt = endpoint(env, prefix + j + 1)
            if prefix + j + 1 in (16, 56):
                render_anchor(env, prefix + j + 1, sample, receipt, output)
            records.append(
                dict(
                    index=j,
                    seed_command=solved["command"],
                    refinement=refined,
                    tcp_error=pose_error(generated[j], source_pose(obs) @ h),
                    limited_history_after=limited_history.innovation.copy(),
                    command=command,
                    state_sha256=receipt["sim_state_sha256"],
                    sim_state=sample["sim_states"],
                    opening=aperture(obs),
                    opening_target=chunk["openings"][j],
                    controller_goal_ori=env.robots[0].controller.goal_ori.copy(),
                    controller_goal_pos=env.robots[0].controller.goal_pos.copy(),
                    motor=env.robots[0].gripper.current_action.copy(),
                    **object_progress(env),
                )
            )
            previous, previous_grip = command.copy(), update.command
            result["branch_steps"] += 1
        result.update(
            complete=True,
            passed=True,
            native_goals_verified=True,
            scope="single saved chunk counterfactual; not complete task or model qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        write_json(output / "steps.json", records)
        write_json(output / "result.json", result)
    print(json.dumps(reference(output / "result.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--horizon", type=int, choices=(1, 40), required=True)
    args = parser.parse_args()
    execute(args.case_index, args.horizon)
