"""Reconstruct causal refinement history and dispatch a recovered tracking action."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/dependency_safe_tracking_dispatch_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    source = REPO / "outputs/tracking_preview_wam_development_v3_20261027/development/cases/task0_init0_s43_terminal058"
    folder = source / "simulation/8400_masked_position_hold/task0_init0_s43"
    manifest_ref = reference(folder / "manifest.json")
    manifest = read(manifest_ref)
    control_ref = reference(folder / "controller.json")
    control = read(control_ref)
    rows_ref = reference(folder / "steps.jsonl")
    rows = [json.loads(x) for x in checked(rows_ref).read_text().splitlines()]
    receipt_ref = reference(source / "endpoints.jsonl")
    receipts = [json.loads(x) for x in checked(receipt_ref).read_text().splitlines()]
    require(len(rows) == 210 and len(receipts) == 211, "Exact failed prefix")
    chunks_ref = reference(folder / "chunks.json")
    chunks = read(chunks_ref)
    require(len(chunks) == 27 and chunks[-1]["step"] == 208, "Last generated chunk before failure")
    root.mkdir()
    result = dict(
        complete=False,
        passed=False,
        source_ref=reference(Path(__file__)),
        manifest_ref=manifest_ref,
        controller_ref=control_ref,
        rows_ref=rows_ref,
        receipt_ref=receipt_ref,
        chunks_ref=chunks_ref,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
    )
    env = None
    try:
        guard = read(CONTROL_REF)["cv2_guard_registration"]
        restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
        from collect_wam_native_training import endpoint, render_anchor

        from s2a.cameras import render_current_cameras
        from s2a.dependency_safe_preview_runtime import solve_preview
        from s2a.geometry import inverse
        from s2a.gripper_motor_state import preserve_motor_target
        from s2a.libero_simulation import current_observation, make_environment, measured_twist

        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        import mujoco

        from s2a.current_coupled_state import read_current_dynamics
        from s2a.tracking_preview_transaction import TrackingPreviewTransaction

        transaction = TrackingPreviewTransaction()
        robot = env.robots[0]
        controller = robot.controller
        current_observation(env)
        robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
        for i in range(211):
            sample, receipt = endpoint(env, i)
            require(receipt["sim_state_sha256"] == receipts[i]["sim_state_sha256"], f"Original prefix endpoint {i}")
            if i in (0, 16, 56):
                render_anchor(env, i, sample, receipt, root)
            if i == 210:
                break
            if i % 8 == 0:
                render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
            row = rows[i]
            x = row["qualified_forward_input"]
            f = row["tracking_refinement"]
            np.testing.assert_array_equal(f["command"], row["command"][:6])
            np.testing.assert_allclose(
                transaction.history.current(x["x_site_pose"][0], x["x_site_twist"][0]),
                f["history_before"],
                atol=1e-12,
                rtol=0,
            )
            transaction.history.stage_plan(f["planned_commands"])
            transaction.history.commit_prediction(
                x["x_site_pose"][0],
                x["x_site_twist"][0],
                f["nominal_response"],
            )
            env.step(np.asarray(row["command"]))
            result["physics_steps"] += 1
            current_observation(env)
            actual_site = np.eye(4)
            actual_site[:3, :3] = np.asarray(env.sim.data.get_site_xmat(controller.eef_name))
            actual_site[:3, 3] = np.asarray(env.sim.data.get_site_xpos(controller.eef_name))
            history_after = transaction.history.observe_completed(actual_site, measured_twist(env))
            np.testing.assert_allclose(history_after, row["tracking_history_after"], atol=1e-12, rtol=0)
        require(transaction.history.completed_periods == 210, "Reconstructed completed causal history")
        model, data = env.sim.model._model, env.sim.data._data
        dofs = np.asarray(robot._ref_joint_vel_indexes)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        pose = np.eye(4)
        pose[:3, 3] = data.site_xpos[sid]
        pose[:3, :3] = data.site_xmat[sid].reshape(3, 3)
        template = read(reference(Path(manifest["runtime_config"]["tool_contract"])))
        tool = np.asarray(template["source_from_tcp"])
        mapping = np.asarray(control["source_from_site"])
        chunk = chunks[-1]
        generated = np.asarray(chunk["effective_targets_tcp"]) @ inverse(tool) @ mapping
        anchor = np.asarray(chunk["anchor_tcp"]) @ inverse(tool) @ mapping
        inputs = dict(
            mass=mass[np.ix_(dofs, dofs)],
            jacobian=np.vstack((jp, jr))[:, dofs],
            pose=pose,
            twist=measured_twist(env),
            generated_poses=generated,
            query_anchor=anchor,
            previous_command=np.asarray(rows[-1]["command"])[:6],
            innovation=np.asarray(rows[-1]["coupled_response_history"]["innovation_after"]),
            sim_state=env.sim.get_state().flatten(),
        )
        with (root / "failing_inputs.npz").open("xb") as stream:
            np.savez_compressed(stream, **inputs)
        from coupled_pose_preview_independent import verify_preview

        solved = solve_preview(
            inputs["mass"],
            inputs["jacobian"],
            pose,
            inputs["twist"],
            generated,
            anchor,
            2,
            inputs["previous_command"],
            control["contract"],
            innovation=inputs["innovation"],
        )
        require(
            solved["preview"]["qp_recovery"]["route"] == "nullspace_after_dependent_blocker",
            "Exact captured failure uses guarded recovery",
        )
        proof = verify_preview(
            solved["preview"],
            pose,
            inputs["twist"],
            np.asarray(solved["references"]),
            np.asarray(solved["reference_twists"]),
            inputs["previous_command"],
            inputs["mass"],
            inputs["jacobian"],
            control["contract"],
            innovation=inputs["innovation"],
        )
        prior_ref = reference(REPO / "outputs/dependency_safe_preview_seed_20261027/verified.json")
        np.testing.assert_array_equal(solved["command"], read(prior_ref)["command"])
        command = np.r_[solved["command"], np.asarray(rows[-1]["command"])[6]]
        require(command.shape == (7,) and np.isfinite(command).all(), "Finite full action")
        require(np.max(abs(command[:6])) <= 1 + 1e-12, "Arm interface bounds")
        require(np.max(abs(command[:6] - inputs["previous_command"])) <= 0.25 + 1e-12, "Arm rate bound")
        current_data = dict(
            x_joint_position=[np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes]],
            x_joint_velocity=[np.asarray(env.sim.data.qvel)[robot._ref_joint_vel_indexes]],
            x_site_pose=[pose],
            x_site_twist=[inputs["twist"]],
            x_previous_goal_ori=[np.asarray(controller.goal_ori).copy()],
        )
        dynamics = read_current_dynamics(env, current_data, command)
        refinement = transaction.prepare(
            env,
            dynamics,
            pose,
            inputs["twist"],
            generated,
            anchor,
            2,
            solved["preview"]["planned_commands"],
            inputs["previous_command"],
            control["contract"],
            site_from_tcp=inverse(mapping) @ tool,
        )
        command[:6] = refinement["command"]
        require(np.isfinite(command).all() and np.max(abs(command[:6])) <= 1 + 1e-12, "Selected action bounds")
        require(np.max(abs(command[:6] - inputs["previous_command"])) <= 0.25 + 1e-12, "Selected rate bound")
        transaction.commit(command[:6])
        before = env.sim.get_state().flatten().copy()
        env.step(command)
        result["physics_steps"] += 1
        current_observation(env)
        np.testing.assert_allclose(controller.goal_pos, refinement["expected_goal_pos"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(controller.goal_ori, refinement["expected_goal_ori"], atol=1e-8, rtol=0)
        actual_site = np.eye(4)
        actual_site[:3, :3] = np.asarray(env.sim.data.get_site_xmat(controller.eef_name))
        actual_site[:3, 3] = np.asarray(env.sim.data.get_site_xpos(controller.eef_name))
        transaction.observe(actual_site, measured_twist(env))
        require(
            transaction.phase == "ready" and transaction.history.completed_periods == 211,
            "Actual completed transaction",
        )
        _, after_receipt = endpoint(env, 211)
        after = env.sim.get_state().flatten().copy()
        require(np.isfinite(after).all() and after[0] > before[0], "Actual finite native time advance")
        with (root / "dispatch.npz").open("xb") as stream:
            np.savez_compressed(stream, before=before, after=after, command=command)
        result.update(
            complete=True,
            passed=True,
            all_prefix_endpoint_sha_equal=True,
            prefix_physics_steps=210,
            new_physics_steps=1,
            captured_seed_proof_ref=prior_ref,
            independent_proof=proof,
            qp_recovery=solved["preview"]["qp_recovery"],
            dispatch_ref=reference(root / "dispatch.npz"),
            after_receipt=after_receipt,
            gripper_policy="Hold previous actual command for this isolated one-step probe",
            tracking_route=refinement["route"],
            completed_history_periods=transaction.history.completed_periods,
            scope="Recovered tracking transaction at one state; gripper held; no task or WAM qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        publish(root / "result.json", result)
    print(json.dumps(reference(root / "result.json")), flush=True)


if __name__ == "__main__":
    main()
