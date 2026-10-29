"""Execute one original correct-reference profile, with explicit baseline/candidate policy."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, write_json
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def numeric(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(18), required=True)
    parser.add_argument("--policy", choices=("baseline", "candidate"), required=True)
    args = parser.parse_args()
    strict_native_environment()
    prepared_ref = reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json")
    prepared = read(prepared_ref)
    require(prepared["passed"], "Prepared references")
    for path, digest in prepared["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    entry = prepared["profiles"][args.case_index]
    profile = entry["profile"]
    root = REPO / f"outputs/tracking_correct_reference_20261027/case{args.case_index:02d}_{args.policy}"
    root.mkdir(exist_ok=False)
    manifest = read(entry["manifest_ref"])
    base = read(entry["execution_config_ref"])
    old_control = read(entry["controller_ref"])
    parameters = read(base["response_parameters_ref"])
    with np.load(checked(profile["trajectory_ref"]), allow_pickle=False) as source:
        poses = source["source_pose"][1:].copy() if profile["format"] == "native_endpoints" else source["poses"].copy()
        openings = (
            source["opening"][1:].copy() if profile["format"] == "native_endpoints" else source["openings"].copy()
        )
    tool_source = np.asarray(read(entry["tool_ref"])["source_from_tcp"])
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    import mujoco
    from collect_wam_native_training import endpoint, render_anchor, save_arrays

    from s2a.action_reference import decode_action_targets, encode_oracle_targets
    from s2a.adaptive_plan_history import AdaptivePlanHistory
    from s2a.checkpoint_simulation import object_progress
    from s2a.coupled_pose_preview_runtime import solve_preview
    from s2a.coupled_response_history import CoupledResponseHistory
    from s2a.current_coupled_response import build_response
    from s2a.current_coupled_state import rotation_goal_from_command
    from s2a.geometry import inverse, pose_error, transform
    from s2a.gripper_motor_state import preserve_motor_target
    from s2a.libero_simulation import aperture, current_observation, make_environment, measured_twist, source_pose
    from s2a.loaded_velocity_control import LoadedVelocityControl
    from s2a.native_training import pack_gripper_caches, synchronous_labels
    from s2a.osc import OSCInterface
    from s2a.task_acceptance import task_outcome
    from s2a.tracking_margin_native import refine_native_preview

    result = dict(
        complete=False,
        passed=False,
        case_index=args.case_index,
        policy=args.policy,
        prepared_ref=prepared_ref,
        source_ref=reference(Path(__file__)),
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
    )
    env = None
    rows = []
    samples = []
    receipts = []
    images = []
    chunks = []
    try:
        env, _ = make_environment(manifest["runtime_config"], root / "native", manifest["record"], 0)
        (root / "model.xml").write_text(env.sim.model.get_xml())
        result["model_ref"] = reference(root / "model.xml")
        result["method_contract"] = dict(
            policy=args.policy,
            source="correct_reference",
            gripper="unchanged LoadedVelocityControl",
            first_caps="reported only in candidate primary stage",
            task_thresholds="stable40;position<=.01;rotation<=.05",
            execution_config_ref=entry["execution_config_ref"],
            controller_ref=entry["controller_ref"],
            tool_ref=entry["tool_ref"],
        )
        robot = env.robots[0]
        controller = robot.controller
        obs = current_observation(env)
        robot.gripper.current_action = preserve_motor_target(robot.gripper.current_action)
        grip = LoadedVelocityControl(parameters, aperture(obs), velocity_envelope=base["gripper_velocity_envelope"])
        interface = OSCInterface(
            *(old_control["contract"][k] for k in ("input_min", "input_max", "output_min", "output_max"))
        )
        model, data = env.sim.model._model, env.sim.data._data
        dofs, qids = np.asarray(robot._ref_joint_vel_indexes), np.asarray(robot._ref_joint_pos_indexes)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, controller.eef_name)
        mapping = np.asarray(old_control["source_from_site"])
        tool = inverse(mapping) @ tool_source
        history = CoupledResponseHistory()
        limited = AdaptivePlanHistory()
        previous = np.zeros(7)
        previous_grip = 0.0

        def observe(index):
            sample, receipt = endpoint(env, index)
            samples.append(sample)
            receipts.append(receipt)
            with (root / "endpoints.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt) + "\n")
            if index in (0, 16, 56):
                images.append(render_anchor(env, index, sample, receipt, root))
            return sample, receipt

        _, first = observe(0)
        require(first["sim_state_sha256"] == entry["initial_state_sha256"], "Original initial state")
        for step in range(profile["physics_steps"]):
            if step % 8 == 0:
                cursor = 0
                anchor = source_pose(obs) @ tool_source
                raw = encode_oracle_targets(anchor, poses[step : step + 40] @ tool_source, openings[step : step + 40])
                targets, generated_openings = decode_action_targets(anchor, raw, horizon=len(raw))
                generated_site = targets @ inverse(tool)
                anchor_site = anchor @ inverse(tool)
                chunks.append(
                    dict(
                        step=step,
                        anchor_tcp=anchor,
                        raw128=raw,
                        effective_targets_tcp=targets,
                        openings=generated_openings,
                    )
                )
                write_json(root / "chunks.json", chunks)
            else:
                cursor = step % 8
            measured = source_pose(obs)
            tcp = measured @ tool_source
            opening_before = aperture(obs)
            motor_before = robot.gripper.current_action.copy()
            finger_velocity = np.asarray(obs["robot0_gripper_qvel"]).copy()
            grip_result = grip.update(
                aperture(obs),
                robot.gripper.current_action,
                0.05 * (finger_velocity[0] - finger_velocity[1]) / 0.079,
                targets,
                generated_openings,
                cursor,
                tcp,
                previous_command=previous_grip,
                dt=0.05,
            )
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, mass, data.qM)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, sid)
            dynamics = dict(
                mass=mass[np.ix_(dofs, dofs)],
                jacobian=np.vstack((jp, jr))[:, dofs],
                robot_joint_position=data.qpos[qids].copy(),
                robot_joint_velocity=data.qvel[dofs].copy(),
                kp=controller.kp.copy(),
                kd=controller.kd.copy(),
            )
            site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
            twist = measured_twist(env)
            coupled_before = history.current(site, twist)
            solved = solve_preview(
                dynamics["mass"],
                dynamics["jacobian"],
                site,
                twist,
                generated_site,
                anchor_site,
                cursor,
                previous[:6],
                old_control["contract"],
                innovation=coupled_before,
            )
            refinement = None
            if args.policy == "candidate":
                refinement = refine_native_preview(
                    env,
                    dynamics,
                    site,
                    twist,
                    generated_site,
                    anchor_site,
                    cursor,
                    solved["preview"]["planned_commands"],
                    previous[:6],
                    old_control["contract"],
                    limited,
                    site_from_tcp=tool,
                )
                arm = refinement["command"]
                limited.commit_prediction(site, twist, refinement["nominal_response"])
                goal = refinement["expected_goal_ori"]
                goal_pos = refinement["expected_goal_pos"]
            else:
                arm = np.asarray(solved["command"])
                goal, _ = rotation_goal_from_command(arm, interface, site[:3, :3], controller.goal_ori)
                goal_pos = site[:3, 3] + interface.decode(arm)[:3]
            command = np.r_[arm, grip_result.command]
            relative = goal @ site[:3, :3].T
            forward = build_response(dynamics["mass"], dynamics["jacobian"], controller.kp, controller.kd, relative)
            history.commit_executed(site, twist, forward, arm[:3], old_control["contract"])
            env.step(command)
            result["physics_steps"] += 1
            obs = current_observation(env)
            np.testing.assert_allclose(controller.goal_pos, goal_pos, atol=1e-9, rtol=0)
            np.testing.assert_allclose(controller.goal_ori, goal, atol=1e-8, rtol=0)
            after_site = transform(data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy())
            history.observe_completed(after_site, measured_twist(env))
            if refinement is not None:
                limited.observe_completed(after_site, measured_twist(env))
            sample, receipt = observe(step + 1)
            row = dict(
                step=step,
                reference_index=cursor,
                chunk_index=len(chunks) - 1,
                command=command,
                seed_command=solved["command"],
                refinement=refinement,
                dynamics=dynamics,
                site_pose=site,
                site_twist=twist,
                coupled_innovation_before=coupled_before,
                coupled_innovation_after=history.innovation.copy(),
                opening_before=opening_before,
                motor_target_before=motor_before,
                measured_period_velocity=0.05 * (finger_velocity[0] - finger_velocity[1]) / 0.079,
                gripper=asdict(grip_result),
                measured_source=measured,
                after_source=source_pose(obs),
                desired_tcp=targets[cursor],
                tcp_error=pose_error(targets[cursor], source_pose(obs) @ tool_source),
                opening=aperture(obs),
                controller_goal_pos=controller.goal_pos.copy(),
                controller_goal_ori=controller.goal_ori.copy(),
                state_sha256=receipt["sim_state_sha256"],
                **object_progress(env),
            )
            rows.append(row)
            with (root / "steps.jsonl").open("a") as stream:
                json.dump(row, stream, allow_nan=False, default=numeric)
                stream.write("\n")
            previous, previous_grip = command.copy(), grip_result.command
        arrays = {key: np.stack([s[key] for s in samples]) for key in samples[0] if key != "gripper_current_action"}
        arrays.update(pack_gripper_caches([s["gripper_current_action"] for s in samples]))
        arrays.update(synchronous_labels(arrays["source_pose"], arrays["opening"], tool_source, arrays["sim_time"]))
        arrays_ref = save_arrays(root / "actual_trajectory.npz", arrays)
        errors = np.asarray([r["tcp_error"] for r in rows])
        outcome = task_outcome(
            np.asarray([r["success"] for r in rows], dtype=bool),
            expected_steps=profile["physics_steps"],
            stable_steps=40,
        )
        pos = float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1))))
        rot = float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1))))
        result.update(
            complete=True,
            passed=True,
            arrays_ref=arrays_ref,
            images=images,
            outcome=outcome,
            position_rmse_m=pos,
            so3_rmse_rad=rot,
            performance_qualified=bool(outcome["stable"] and pos <= 0.01 and rot <= 0.05),
            independent_check_pending=True,
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if env is not None:
            if samples and "arrays_ref" not in result:
                partial = {
                    key: np.stack([s[key] for s in samples]) for key in samples[0] if key != "gripper_current_action"
                }
                partial.update(pack_gripper_caches([s["gripper_current_action"] for s in samples]))
                result["partial_arrays_ref"] = save_arrays(root / "partial_trajectory.npz", partial)
            env.close()
        write_json(root / "steps.json", rows)
        write_json(root / "result.json", result)
    print(json.dumps(reference(root / "result.json")), flush=True)


if __name__ == "__main__":
    main()
