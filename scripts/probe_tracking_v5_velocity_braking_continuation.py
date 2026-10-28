"""Pair two causal controllers on the same already-generated40 targets."""

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
    root = REPO / "outputs/tracking_v5_velocity_braking_continuation_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    source = REPO / "outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058"
    worker_ref = reference(source / "worker_result.json")
    worker = read(worker_ref)
    require(worker["complete"] is worker["passed"] is True, "Closed original case")
    folder = Path(worker["physical_case_dir"])
    manifest_ref = reference(folder / "manifest.json")
    manifest = read(manifest_ref)
    control = read(reference(folder / "controller.json"))
    config = read(worker["config_ref"])
    rows_ref = reference(folder / "steps.jsonl")
    rows = [json.loads(x) for x in checked(rows_ref).read_text().splitlines()]
    receipts = [json.loads(x) for x in checked(worker["endpoint_receipts_ref"]).read_text().splitlines()]
    chunks_ref = reference(folder / "chunks.json")
    chunks = read(chunks_ref)
    require(len(rows) == 600 and len(receipts) == 601 and len(chunks) == 75, "Original complete sources")
    proof_ref = reference(REPO / "outputs/tracking_v5_velocity_braking_numeric_20261028/result.json")
    require(read(proof_ref)["complete"] is read(proof_ref)["passed"] is True, "Closed fixed braking numeric probe")
    numeric = read(proof_ref)
    numeric_candidate = next(e for e in numeric["entries"] if e["step"] == 368)["candidate"]
    require(numeric_candidate["witness_found"] is True, "Captured initial braking witness")
    start = 368
    selected_chunk = chunks[start // 8]
    require(selected_chunk["step"] == 368, "Registered already-generated chunk46")
    registration = publish(
        root / "registration.json",
        dict(
            worker_ref=worker_ref,
            manifest_ref=manifest_ref,
            rows_ref=rows_ref,
            chunks_ref=chunks_ref,
            source_ref=reference(Path(__file__)),
            braking_numeric_ref=proof_ref,
            inner_observation="Read q/time after each native sim.step during continuation only; no extra forward/step",
            source_refs=[
                reference(REPO / "src/s2a" / name)
                for name in (
                    "velocity_braking_joint_plan.py",
                    "velocity_braking_tracking_hybrid.py",
                    "velocity_braking_tracking_native.py",
                )
            ],
            seed=42,
            branch_steps=408,
            physics_budget=816,
            braking_time_s=0.1,
            braking_time_policy="Two20Hz control periods, fixed before numeric and physical outcomes",
            prefix_steps=368,
            continuation_steps=40,
            world_model_calls=0,
            fresh_cases=0,
            candidate_policy="Same already-generated query46 references for40 periods; actual feedback and fixed0.1s velocity-braking tracking controller",
            baseline_policy="Same query46 reference for40 periods; actual feedback and original tracking controller",
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        branches=[],
    )
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor

    from s2a.cameras import render_current_cameras
    from s2a.checkpoint_simulation import object_progress
    from s2a.coupled_response_history import CoupledResponseHistory
    from s2a.current_coupled_response import build_response
    from s2a.current_coupled_state import read_current_dynamics
    from s2a.dependency_safe_preview_runtime import solve_preview
    from s2a.geometry import inverse, pose_error
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import aperture, current_observation, make_environment, measured_twist, source_pose
    from s2a.loaded_velocity_release_floor import LoadedVelocityControl
    from s2a.tracking_preview_transaction import TrackingPreviewTransaction
    from s2a.velocity_braking_tracking_native import refine_native_preview as braking_refiner

    tool = np.asarray(read(reference(Path(manifest["runtime_config"]["tool_contract"])))["source_from_tcp"])
    mapping = np.asarray(control["source_from_site"])
    contract = control["contract"]
    env = None
    try:
        for branch in ("original", "braking"):
            env, _ = make_environment(manifest["runtime_config"], root / branch / "native", manifest["record"], 0)
            robot = env.robots[0]
            controller = robot.controller
            obs = current_observation(env)
            robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
            grip = LoadedVelocityControl(
                read(config["response_parameters_ref"]),
                aperture(obs),
                velocity_envelope=config["gripper_velocity_envelope"],
            )
            transaction = TrackingPreviewTransaction(refiner=braking_refiner if branch == "braking" else None)
            coupled = CoupledResponseHistory()
            limits = np.asarray(
                [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
            )
            inner = []
            native_step = env.sim.step

            def observed_step(*args, **kwargs):
                native_step(*args, **kwargs)
                inner.append(
                    dict(
                        time=float(env.sim.data.time),
                        q=np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes].copy(),
                    )
                )

            previous = np.zeros(7)
            log = []
            states = []
            for i in range(409):
                sample, receipt = endpoint(env, i)
                states.append(env.sim.get_state().flatten().copy())
                if i <= start:
                    require(
                        receipt["sim_state_sha256"] == receipts[i]["sim_state_sha256"], f"{branch} exact endpoint{i}"
                    )
                if i in (0, 16, 56):
                    render_anchor(env, i, sample, receipt, root / branch)
                if i == 408:
                    break
                if i % 8 == 0:
                    render_current_cameras(env, manifest["runtime_config"]["render_resolution"])
                active = i >= start
                chunk = selected_chunk if active else chunks[i // 8]
                cursor = i - selected_chunk["step"] if active else i % 8
                targets = np.asarray(chunk["effective_targets_tcp"])
                openings = np.asarray(chunk["openings"])
                anchor = np.asarray(chunk["anchor_tcp"]) @ inverse(tool) @ mapping
                generated = targets @ inverse(tool) @ mapping
                site = np.eye(4)
                site[:3, :3] = np.asarray(env.sim.data.get_site_xmat(controller.eef_name))
                site[:3, 3] = np.asarray(env.sim.data.get_site_xpos(controller.eef_name))
                twist = measured_twist(env)
                velocity = np.asarray(obs["robot0_gripper_qvel"])
                g = grip.update(
                    aperture(obs),
                    robot.gripper.current_action,
                    0.05 * (velocity[0] - velocity[1]) / 0.079,
                    targets,
                    openings,
                    cursor,
                    source_pose(obs) @ tool,
                    previous_command=float(previous[6]),
                    dt=0.05,
                )
                extra = {}
                if active:
                    current_data = dict(
                        x_joint_position=[np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes]],
                        x_joint_velocity=[np.asarray(env.sim.data.qvel)[robot._ref_joint_vel_indexes]],
                        x_site_pose=[site],
                        x_site_twist=[twist],
                        x_previous_goal_ori=[np.asarray(controller.goal_ori).copy()],
                    )
                    provisional = np.r_[previous[:6], g.command]
                    dynamics = read_current_dynamics(env, current_data, provisional)
                    innovation = coupled.current(site, twist)
                    solved = solve_preview(
                        dynamics["mass"],
                        dynamics["jacobian"],
                        site,
                        twist,
                        generated,
                        anchor,
                        cursor,
                        previous[:6],
                        contract,
                        innovation=innovation,
                    )
                    refinement = transaction.prepare(
                        env,
                        dynamics,
                        site,
                        twist,
                        generated,
                        anchor,
                        cursor,
                        solved["preview"]["planned_commands"],
                        previous[:6],
                        contract,
                        site_from_tcp=inverse(mapping) @ tool,
                    )
                    command = np.r_[refinement["command"], g.command]
                    transaction.commit(command[:6])
                    executed = read_current_dynamics(env, current_data, command)
                    model = build_response(
                        executed["mass"],
                        executed["jacobian"],
                        executed["kp"],
                        executed["kd"],
                        executed["relative_rotation_goal"],
                    )
                    coupled.commit_executed(site, twist, model, command[:3], contract)
                    extra = dict(seed=solved, refinement=refinement, dynamics=executed)
                else:
                    row = rows[i]
                    np.testing.assert_allclose(g.command, row["command"][6], atol=1e-12, rtol=0)
                    command = np.asarray(row["command"])
                    f = row["tracking_refinement"]
                    np.testing.assert_allclose(
                        transaction.history.current(site, twist), f["history_before"], atol=1e-12, rtol=0
                    )
                    transaction.history.stage_plan(f["planned_commands"])
                    transaction.history.commit_prediction(site, twist, f["nominal_response"])
                    coupled.current(site, twist)
                    coupled.pending = (
                        site.copy(),
                        np.asarray(row["coupled_response_history"]["commit"]["nominal_response"]),
                    )
                if i == start:
                    np.testing.assert_allclose(
                        solved["preview"]["planned_commands"],
                        rows[i]["pose_preview_seed_inverse"]["preview"]["planned_commands"],
                        atol=1e-8,
                        rtol=0,
                    )
                    expected = (
                        rows[i]["command"]
                        if branch == "original"
                        else np.r_[numeric_candidate["planned_commands"][0], g.command]
                    )
                    np.testing.assert_allclose(command, expected, atol=1e-7, rtol=0)
                require(np.isfinite(command).all() and np.max(abs(command)) <= 1 + 1e-12, "Finite feasible action")
                require(np.max(abs(command[:6] - previous[:6])) <= 0.25 + 1e-12, "Arm rate bound")
                if active:
                    inner.clear()
                    env.sim.step = observed_step
                env.step(command)
                result["physics_steps"] += 1
                if active:
                    env.sim.step = native_step
                    require(len(inner) == 25, "All25 actual2ms inner samples")
                    np.testing.assert_allclose(np.diff([v["time"] for v in inner]), 0.002, atol=1e-12, rtol=0)
                obs = current_observation(env)
                after_site = np.eye(4)
                after_site[:3, :3] = np.asarray(env.sim.data.get_site_xmat(controller.eef_name))
                after_site[:3, 3] = np.asarray(env.sim.data.get_site_xpos(controller.eef_name))
                after_twist = measured_twist(env)
                if active:
                    np.testing.assert_allclose(controller.goal_pos, refinement["expected_goal_pos"], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(controller.goal_ori, refinement["expected_goal_ori"], atol=1e-8, rtol=0)
                    transaction.observe(after_site, after_twist)
                else:
                    h = transaction.history.observe_completed(after_site, after_twist)
                    np.testing.assert_allclose(h, rows[i]["tracking_history_after"], atol=1e-12, rtol=0)
                innovation_after = coupled.observe_completed(after_site, after_twist)
                if not active:
                    np.testing.assert_allclose(
                        innovation_after, rows[i]["coupled_response_history"]["innovation_after"], atol=1e-12, rtol=0
                    )
                if i >= start:
                    progress = object_progress(env)
                    log.append(
                        dict(
                            step=i,
                            cursor=cursor,
                            command=command.copy(),
                            opening=aperture(obs),
                            success=progress["success"],
                            objects=progress["objects"],
                            tcp_error=pose_error(targets[cursor], source_pose(obs) @ tool),
                            actual_inner=list(inner),
                            actual_joint_after=np.asarray(env.sim.data.qpos)[robot._ref_joint_pos_indexes].copy(),
                            **extra,
                        )
                    )
                previous = command.copy()
            require(
                len(log) == 40 and sum(len(r["actual_inner"]) for r in log) == 1000,
                "Full40 periods and1000 inner samples",
            )
            states_path = root / branch / "states.npz"
            with states_path.open("xb") as stream:
                np.savez_compressed(stream, sim_states=np.asarray(states))
            steps_ref = publish(root / branch / "steps.json", log)
            result["branches"].append(
                dict(
                    branch=branch,
                    steps_ref=steps_ref,
                    states_ref=reference(states_path),
                    prefix_steps=368,
                    continuation_steps=40,
                    inner_ref=publish(
                        root / branch / "inner_limits.json",
                        dict(
                            joint_limits=limits,
                            minimum_margin_rad=float(
                                np.min(
                                    [
                                        np.minimum(
                                            np.asarray(v["q"]) - limits[:, 0], limits[:, 1] - np.asarray(v["q"])
                                        ).min()
                                        for row in log
                                        for v in row["actual_inner"]
                                    ]
                                )
                            ),
                            samples=sum(len(row["actual_inner"]) for row in log),
                        ),
                    ),
                    success_steps=sum(bool(r["success"]) for r in log),
                    grasp_steps=sum(bool(r["objects"]["akita_black_bowl_1"]["grasp_contact"]) for r in log),
                    terminal=bool(log[-1]["success"]),
                    all40_success=all(bool(r["success"]) for r in log),
                )
            )
            env.close()
            env = None
        require(result["physics_steps"] == 816, "Exact two408-period branches")
        result.update(
            complete=True,
            passed=True,
            scope="Paired local single-generated-chunk continuation; no responsive WAM or full task/fresh qualification",
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if env is not None:
            env.close()
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
