"""Independently replay committed-chunk commands and recompute actual local outcomes."""

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
    root = REPO / "outputs/committed_generated_chunk_check_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    producer_ref = reference(REPO / "outputs/committed_generated_chunk_20261027/result.json")
    producer = read(producer_ref)
    require(producer["complete"] is producer["passed"] is True, "Complete paired producer")
    reg = read(producer["registration_ref"])
    manifest = read(reg["manifest_ref"])
    original = [json.loads(x) for x in checked(reg["rows_ref"]).read_text().splitlines()]
    candidate = producer["branches"][1]
    rows = read(candidate["steps_ref"])
    require(candidate["branch"] == "committed" and len(rows) == 40, "Full committed continuation")
    chunk = read(reg["chunks_ref"])[17]
    require(chunk["step"] == 136, "Registered chunk")
    registration = publish(
        root / "registration.json",
        dict(
            producer_ref=producer_ref,
            source_ref=reference(Path(__file__)),
            physics_budget=176,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    summary = dict(
        complete=False, passed=False, registration_ref=registration, physics_steps=0, world_model_calls=0, fresh_cases=0
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.geometry import pose_error
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import aperture, current_observation, make_environment, source_pose

    tool = np.asarray(read(reference(Path(manifest["runtime_config"]["tool_contract"])))["source_from_tcp"])
    env = None
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        robot = env.robots[0]
        obs = current_observation(env)
        robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
        controller = robot.controller
        joint_ids = [env.sim.model.joint_name2id(f"robot0_joint{k}") for k in range(1, 8)]
        limits = np.asarray(env.sim.model.jnt_range)[joint_ids]
        errors = []
        margins = []
        success = []
        grasp = []
        maximum_state_difference = 0.0
        with np.load(checked(candidate["states_ref"]), allow_pickle=False) as arrays:
            require(arrays["sim_states"].shape[0] == 177, "All177 saved endpoints")
            for i in range(177):
                sample, receipt = endpoint(env, i)
                state = env.sim.get_state().flatten()
                np.testing.assert_allclose(state, arrays["sim_states"][i], atol=1e-8, rtol=0)
                maximum_state_difference = max(
                    maximum_state_difference, float(np.max(abs(state - arrays["sim_states"][i])))
                )
                q = np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes]
                margins.append(np.minimum(q - limits[:, 0], limits[:, 1] - q))
                if i in (0, 16, 56):
                    render_anchor(env, i, sample, receipt, root)
                if i == 176:
                    break
                if i % 8 == 0:
                    render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
                command = np.asarray(original[i]["command"] if i < 136 else rows[i - 136]["command"])
                env.step(command)
                summary["physics_steps"] += 1
                obs = current_observation(env)
                if i >= 136:
                    row = rows[i - 136]
                    progress = object_progress(env)
                    require(progress["success"] == row["success"], "Independent native task predicate")
                    require(
                        progress["objects"]["akita_black_bowl_1"]["grasp_contact"]
                        == row["objects"]["akita_black_bowl_1"]["grasp_contact"],
                        "Independent target grasp predicate",
                    )
                    error = pose_error(np.asarray(chunk["effective_targets_tcp"])[i - 136], source_pose(obs) @ tool)
                    np.testing.assert_allclose(error, row["tcp_error"], atol=1e-8, rtol=0)
                    np.testing.assert_allclose(aperture(obs), row["opening"], atol=1e-8, rtol=0)
                    np.testing.assert_allclose(
                        controller.goal_pos, row["refinement"]["expected_goal_pos"], atol=1e-9, rtol=0
                    )
                    np.testing.assert_allclose(
                        controller.goal_ori, row["refinement"]["expected_goal_ori"], atol=1e-8, rtol=0
                    )
                    errors.append(error)
                    success.append(progress["success"])
                    grasp.append(progress["objects"]["akita_black_bowl_1"]["grasp_contact"])
        errors = np.asarray(errors)
        margins = np.asarray(margins)[136:]
        summary.update(
            complete=True,
            passed=True,
            maximum_state_difference=maximum_state_difference,
            success_steps=sum(success),
            grasp_steps=sum(grasp),
            terminal=bool(success[-1]),
            all40_success=all(success),
            position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            so3_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            minimum_actual_joint_margin_rad=margins.min(axis=0),
            outside_joint_endpoint_counts=(margins < 0).sum(axis=0),
            errors=errors,
            scope="Independent176-period native command replay and local40 outcomes; no full600/WAM/fresh qualification",
        )
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        publish(root / "verified.json", summary)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
