"""Fixed saved-reference joint-limit candidate with original prefix and native goals."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, write_json
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF

ROOT = REPO / "outputs/long_joint_physical_v2_branches_20261027"
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
        from s2a.geometry import pose_error
        from s2a.gripper_motor_state import preserve_motor_target
        from s2a.libero_simulation import aperture, current_observation, make_environment, source_pose
        from s2a.loaded_velocity_control import LoadedVelocityControl

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
        plan_verified_ref = reference(REPO / "outputs/long_joint_preview_verified_20261027/verified.json")
        plan_verified = read(plan_verified_ref)
        require(plan_verified["passed"] is plan_verified["complete"] is True, "Independent plan check required")
        producer = read(plan_verified["producer_ref"])
        registration = read(producer["registration_ref"])
        for path, digest in registration["source_sha256"].items():
            checked(dict(path=path, sha256=digest))
        plan = next(r for r in producer["cases"][index]["results"] if r["horizon"] == horizon)
        commands = np.asarray(plan["planned_commands"])
        result["plan_verified_ref"] = plan_verified_ref
        chunk = chunks[case["event"]["query_index"]]
        generated = np.asarray(chunk["effective_targets_tcp"])
        site_from_tcp = np.linalg.inv(mapping) @ h
        from s2a.current_coupled_state import rotation_goal_from_command
        from s2a.osc import OSCInterface

        interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
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
            site_before = np.eye(4)
            site_before[:3, :3] = env.sim.data.get_site_xmat(env.robots[0].controller.eef_name)
            site_before[:3, 3] = env.sim.data.get_site_xpos(env.robots[0].controller.eef_name)
            delta = interface.decode(commands[j])
            require(np.any(delta[3:] != 0), "Unsupported zero rotation latch")
            expected_pos = site_before[:3, 3] + delta[:3]
            expected_ori, _ = rotation_goal_from_command(
                commands[j], interface, site_before[:3, :3], env.robots[0].controller.goal_ori
            )
            command = np.r_[commands[j], update.command]
            env.step(command)
            result["branch_steps"] += 1
            obs = current_observation(env)
            np.testing.assert_allclose(env.robots[0].controller.goal_pos, expected_pos, atol=1e-9, rtol=0)
            np.testing.assert_allclose(env.robots[0].controller.goal_ori, expected_ori, atol=1e-8, rtol=0)
            sample, receipt = endpoint(env, prefix + j + 1)
            tcp = source_pose(obs) @ h
            predicted_tcp = np.asarray(plan["rollout"]["pose"])[j] @ site_from_tcp
            records.append(
                dict(
                    index=j,
                    command=command,
                    tcp=tcp,
                    tcp_error=pose_error(generated[j], tcp),
                    prediction_error=pose_error(predicted_tcp, tcp),
                    sim_state=sample["sim_states"],
                    state_sha256=receipt["sim_state_sha256"],
                    opening=aperture(obs),
                    opening_target=chunk["openings"][j],
                    controller_goal_pos=env.robots[0].controller.goal_pos.copy(),
                    controller_goal_ori=env.robots[0].controller.goal_ori.copy(),
                    **object_progress(env),
                )
            )
            previous_grip = update.command
        result.update(
            complete=True,
            passed=True,
            native_goals_verified=True,
            scope="saved open-loop plan native dynamics audit only; not a qualified closed-loop policy",
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
    parser.add_argument("--horizon", type=int, choices=(8, 16), required=True)
    args = parser.parse_args()
    execute(args.case_index, args.horizon)
