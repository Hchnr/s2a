"""Factorial LIBERO diagnostics with explicit pre/post-action label timing.

The published HDF5 stores states[t] before actions[t] and obs[t] after it.
Oracle references below are post-action observations in the original world
frame. Only native/ceiling diagnostics may send stored expert commands.
"""

import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.geometry import advance, pose_error, transform
from s2a.gripper import OpeningIntent, OpeningLatch, opening_position_target
from s2a.libero_simulation import aperture, controller_contract, make_environment, measured_twist, source_pose
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.perturbations import ActuatorDelay


def post_action_reference(positions, rotation_vectors, gripper_positions):
    """One endpoint per native action; no skip, reanchoring or future restore."""
    positions, rotation_vectors, gripper_positions = map(np.asarray, (positions, rotation_vectors, gripper_positions))
    n = len(positions)
    if positions.shape != (n, 3) or rotation_vectors.shape != (n, 3) or gripper_positions.shape != (n, 2) or n == 0:
        raise ValueError("inconsistent post-action reference shapes")
    if not all(np.isfinite(a).all() for a in (positions, rotation_vectors, gripper_positions)):
        raise ValueError("nonfinite expert reference")
    poses = transform(positions, Rotation.from_rotvec(rotation_vectors).as_matrix())
    opening = np.clip((gripper_positions[:, 0] - gripper_positions[:, 1] - 0.001) / 0.079, 0, 1)
    return poses, opening


def contact_counts(env):
    """Physical contacts involving the target bowl; counts, not force estimates."""
    counts = dict(finger_bowl=0, bowl_plate=0, bowl_other=0)
    for contact in env.sim.data.contact[: env.sim.data.ncon]:
        names = [env.sim.model.geom_id2name(int(i)) or "" for i in (contact.geom1, contact.geom2)]
        if contact.dist > 0 or not any("akita_black_bowl" in n for n in names):
            continue
        if any("finger" in n for n in names):
            counts["finger_bowl"] += 1
        elif any("plate_" in n for n in names):
            counts["bowl_plate"] += 1
        else:
            counts["bowl_other"] += 1
    return counts


def run_audit_case(config, output, case, trial):
    import h5py

    sim_config = {**config["simulation"], "camera_observations": False}
    fixture = json.loads(Path(sim_config["fixture"]).read_text())
    record = fixture["episodes"][trial]
    dest = output / case["name"] / f"trial{trial}"
    dest.mkdir(parents=True, exist_ok=False)
    with h5py.File(fixture["source_hdf5"]) as source:
        demo = source["data"][record["demo"]]
        native = demo["actions"][:]
        states = demo["states"][:]
        grip_labels = demo["obs/gripper_states"][:]
        poses, openings = post_action_reference(demo["obs/ee_pos"][:], demo["obs/ee_ori"][:], grip_labels)
    if len(native) != len(poses):
        raise ValueError("one post-action reference is required per native action")
    manifest = dict(
        config=config,
        case=case,
        trial=trial,
        code=code_fingerprint(),
        input_sha256={p: sha256(p) for p in (sim_config["fixture"], fixture["source_hdf5"])},
        reference="obs[t] is AFTER actions[t]; states[t] is BEFORE actions[t]",
        evidence="Independent MuJoCo; oracle state references; native commands only in explicit ceiling diagnostics",
        runtime={p: importlib.metadata.version(p) for p in ("mujoco", "robosuite", "numpy", "scipy")},
    )
    write_json(dest / "manifest.json", manifest)
    env, obs = make_environment(sim_config, output, record, trial)
    rows, error, success = [], None, False
    ever_success, first_success_step = False, None
    started = time.monotonic()
    try:
        controller, contract = controller_contract(env)
        initial = source_pose(obs)
        site_pose = transform(
            env.sim.data.get_site_xpos(controller.eef_name),
            env.sim.data.get_site_xmat(controller.eef_name).reshape(3, 3),
        )
        initial_cache_error = pose_error(site_pose, transform(controller.ee_pos, controller.ee_ori_mat))
        if case.get("refresh_controller", True):
            controller.update(force=True)
        gripper = env.robots[0].gripper
        if case["initial_gripper"] == "open":
            gripper.current_action = np.array([1.0, -1.0])
        elif case["initial_gripper"] == "measured":
            gripper.current_action = opening_position_target(aperture(obs))
        elif case["initial_gripper"] != "zero":
            raise ValueError("unknown initial gripper state")
        joint_indices = np.asarray(env.robots[0]._ref_gripper_joint_pos_indexes)
        shift_error = np.max(np.abs(states[1:, 1 + joint_indices] - grip_labels[:-1]))
        write_json(
            dest / "initialization.json",
            dict(
                controller=contract,
                controller_cache_error_before_refresh=initial_cache_error,
                initial_gripper_internal=gripper.current_action,
                initial_opening=aperture(obs),
                initial_source=initial,
                stored_obs0_source=poses[0],
                states_tplus1_vs_obs_t_gripper_max_abs=float(shift_error),
            ),
        )
        converter = OSCStateToAction(
            OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max"))),
            contract["kp"],
            contract["kd"],
            1 / sim_config["control_fps"],
            compensate=case["arm"] == "response",
            inverse_weight=config["inverse_weight"],
            max_command_rate=config["max_command_rate"] if case["arm"] == "response" else None,
        )
        intent = OpeningIntent(aperture(obs), band=config["opening_hysteresis"])
        latch = OpeningLatch(**config["opening_latch"])
        position_target = np.zeros(2)
        delay = ActuatorDelay(
            case.get("delay_steps", 0),
            dimensions=9 if case["gripper"] == "position" else 7,
            initial_command=(np.r_[np.zeros(7), gripper.current_action] if case["gripper"] == "position" else None),
        )
        noise = np.random.default_rng(config["seed"] + trial)
        pose_std = np.array([case.get("position_noise_std_m", 0)] * 3 + [case.get("rotation_noise_std_rad", 0)] * 3)
        velocity_std = np.array(
            [case.get("linear_velocity_noise_std_m_s", 0)] * 3 + [case.get("angular_velocity_noise_std_rad_s", 0)] * 3
        )
        if case["gripper"] == "position":

            def position_action(_):
                gripper.current_action = position_target.copy()
                return gripper.current_action

            gripper.format_action = position_action
        length = len(native) + (
            config["extra_hold_steps"] if case["arm"] != "native" else config.get("native_extra_hold_steps", 0)
        )
        with (dest / "steps.jsonl").open("w") as stream:
            for step in range(length):
                index = min(step, len(poses) - 1)
                current = source_pose(obs)
                feedback = advance(current, noise.normal(size=6) * pose_std, 1)
                feedback_velocity = measured_twist(env) + noise.normal(size=6) * velocity_std
                converted = converter.convert(feedback, poses[index], feedback_velocity)
                if case["arm"] == "native":
                    arm = native[index, :6] if step < len(native) else np.zeros(6)
                else:
                    arm = converted["command"]
                if case["gripper"] == "native":
                    grip_command = float(native[index, 6])
                elif case["gripper"] == "intent":
                    grip_command = intent.update(float(openings[index]))
                elif case["gripper"] == "latch":
                    grip_command = latch.update(float(openings[index]))
                elif case["gripper"] == "position":
                    position_target[:] = opening_position_target(float(openings[index]))
                    grip_command = 0.0
                else:
                    raise ValueError("unknown gripper mode")
                issued = np.r_[arm, grip_command]
                issued_position_target = position_target.copy() if case["gripper"] == "position" else None
                if case["gripper"] == "position":
                    delayed = delay.update(np.r_[issued, position_target])
                    command, position_target[:] = delayed[:7], delayed[7:]
                else:
                    command = delay.update(issued)
                before_time = float(env.sim.data.time)
                obs, _, _, _ = env.step(command)
                after = source_pose(obs)
                success = bool(env.check_success())
                if success and first_success_step is None:
                    first_success_step = step
                ever_success |= success
                row = dict(
                    step=step,
                    reference_index=index,
                    time_before=before_time,
                    actual_dt=float(env.sim.data.time) - before_time,
                    measured_source=current,
                    feedback_source=feedback,
                    issued=issued,
                    issued_position_target=issued_position_target,
                    desired_source=poses[index],
                    after_source=after,
                    error=pose_error(poses[index], after),
                    opening_target=float(openings[index]),
                    opening_after=aperture(obs),
                    executed=command,
                    gripper_internal=gripper.current_action.copy(),
                    actuator_control=env.sim.data.ctrl[-2:].copy(),
                    bowl_position=env.sim.data.get_body_xpos("akita_black_bowl_1_main").copy(),
                    plate_position=env.sim.data.get_body_xpos("plate_1_main").copy(),
                    contacts=contact_counts(env),
                    success=success,
                    arm_saturated=converted["saturated"],
                    arm_rate_limited=converted["rate_limited"],
                )
                rows.append(row)
                stream.write(json.dumps(row, default=lambda x: x.tolist(), allow_nan=False) + "\n")
                if success and config.get("stop_on_success", True):
                    break
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
    result = dict(
        case=case,
        trial=trial,
        episode=record["episode"],
        steps=len(rows),
        success=ever_success,
        terminal_success=success,
        first_success_step=first_success_step,
        error=error,
        elapsed_s=time.monotonic() - started,
        seed=config["seed"] + trial,
    )
    if rows:
        errors = np.asarray([r["error"] for r in rows])
        result.update(
            position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            opening_rmse=float(np.sqrt(np.mean([(r["opening_target"] - r["opening_after"]) ** 2 for r in rows]))),
            max_bowl_lift_m=float(max(r["bowl_position"][2] for r in rows) - rows[0]["bowl_position"][2]),
            finger_contact_steps=sum(r["contacts"]["finger_bowl"] > 0 for r in rows),
            plate_contact_steps=sum(r["contacts"]["bowl_plate"] > 0 for r in rows),
            final_second_success_fraction=float(np.mean([r["success"] for r in rows[-20:]])),
        )
    write_json(dest / "result.json", result)
    if error:
        raise RuntimeError(error)
    return result
