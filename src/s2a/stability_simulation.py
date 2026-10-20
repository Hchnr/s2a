"""Instrumented experimental WAM executor, separate from sealed evaluations."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests

from s2a.action_reference import decode_action_targets
from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.cameras import render_current_cameras
from s2a.checkpoint_simulation import object_progress
from s2a.geometry import inverse, pose_error
from s2a.gripper import OpeningIntent, opening_position_target
from s2a.gripper_tracking import GripperTrackingAdapter, GripperTrackingConfig, panda_opening_from_motor_target
from s2a.libero_simulation import (
    aperture,
    controller_contract,
    current_observation,
    generate,
    make_environment,
    measured_twist,
    sha256_bytes,
    source_pose,
)
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.osc_rotation import bounded_osc_rotation_inverse
from s2a.reference_governor import ReferenceGovernor, ReferenceGovernorConfig


def run_stability_case(config, root, record, *, source, variant_name, sampling_seed, checkpoint=None):
    if source not in ("wam", "oracle"):
        raise ValueError("source must be WAM or explicitly privileged oracle qualification")
    variant = next(v for v in config["variants"] if v["name"] == variant_name)
    if variant["rotation"] not in ("response", "bounded") or variant["gripper"] not in ("intent", "tracking"):
        raise ValueError("unknown executor variant")
    template = json.loads(Path(config["policy_template"]).read_text())
    c = dict(
        template,
        seed=sampling_seed,
        task_id=record["task_id"],
        task_text=record["task"],
        camera_observations=False,
        record_video=False,
        return_video=False,
        refresh_controller_after_restore=True,
        action_chunk_size=config["online_chunk_size"],
        sampling_steps=config["sampling_steps"],
        guidance=config["guidance"],
        shift=config["shift"],
        control_fps=config["control_fps"],
    )
    label = f"{checkpoint or source}_{variant_name}"
    dest = root / "simulation" / label / f"{record['name']}_s{sampling_seed}"
    dest.mkdir(parents=True, exist_ok=False)
    code = code_fingerprint()
    critical_files = (
        "scripts/run_wam_stability_matrix.py",
        "src/s2a/stability_simulation.py",
        "src/s2a/reference_governor.py",
        "src/s2a/osc_rotation.py",
        "src/s2a/gripper_tracking.py",
        "src/s2a/checkpoint_simulation.py",
        "src/s2a/action_reference.py",
        "src/s2a/libero_simulation.py",
        "src/s2a/cameras.py",
        "src/s2a/osc.py",
        "src/s2a/gripper.py",
        "src/s2a/geometry.py",
        "src/s2a/artifacts.py",
        config["policy_template"],
    )
    execution_sha256 = {name: sha256(name) for name in critical_files}
    manifest = dict(
        code=code,
        execution_sha256=execution_sha256,
        experiment=config,
        runtime_config=c,
        record=record,
        source=source,
        variant=variant,
        inference_latency_excluded=True,
        oracle_is_wam_success_evidence=False,
        online_control_inputs="current TCP/twist, actuator goal, aperture/history and generated reference only",
    )
    if source == "wam":
        ck = next(ck for ck in config["checkpoints"] if ck["name"] == checkpoint)
        if ck["gpu"] not in (4, 5, 6, 7):
            raise ValueError("only physical GPU4–7 are allowed")
        c["server_url"] = f"http://127.0.0.1:{ck['port']}"
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(c["server_url"] + "/info", timeout=15)
            response.raise_for_status()
            info = response.json()
        expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
        if (
            Path(info["checkpoint_path"]).resolve() != expected.resolve()
            or info["ema_weights"]
            or not info["acceptance_intervention"]["active_slot_mask"]
        ):
            raise ValueError("wrong model or missing inactive-slot mask")
        manifest["server_info"] = info
    else:
        path = Path(config["oracle_reference_root"]) / record["name"] / "reference.npz"
        # The root contains seed-specific native references explicitly selected
        # by fixture name below; no expert file is opened on the WAM branch.
        if not path.exists():
            path = Path(config["oracle_reference_root"]) / f"{record['name']}_s{sampling_seed}" / "reference.npz"
        manifest["oracle_reference_sha256"] = sha256(path)
        with np.load(path) as data:
            reference, reference_opening = data["poses"], data["openings"]
    h = np.asarray(json.loads(Path(c["tool_contract"]).read_text())["source_from_tcp"])
    h_inv = inverse(h)
    manifest["tool_sha256"] = sha256(c["tool_contract"])
    write_json(dest / "manifest.json", manifest)
    env, _ = make_environment(c, dest, record, 0)
    rows, chunks, failure = [], [], None
    start = time.monotonic()
    dt = 1 / c["control_fps"]
    try:
        obs = current_observation(env)
        controller, contract = controller_contract(env)
        if not np.isclose(env.env.control_timestep, dt):
            raise ValueError("control interval differs from the declared model time")
        interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        converter = OSCStateToAction(
            interface,
            contract["kp"],
            contract["kd"],
            dt,
            compensate=True,
            inverse_weight=config["inverse_weight"],
            max_command_rate=config["max_command_rate"],
        )
        initial_opening = aperture(obs)
        env.robots[0].gripper.current_action = opening_position_target(initial_opening)
        grip_config = GripperTrackingConfig(**config["gripper_tracking"])
        gripper = GripperTrackingAdapter(initial_opening, config=grip_config)
        intent = OpeningIntent(initial_opening, band=config["opening_hysteresis"])
        governor = ReferenceGovernor(config=ReferenceGovernorConfig(**config["reference_governor"]))
        write_json(dest / "controller.json", dict(contract=contract, gripper=asdict(grip_config)))
        previous_command = np.zeros(7)
        targets, openings, index, chunk_step, governed = None, None, 0, 0, None
        initial_objects = object_progress(env)["objects"]
        with (dest / "steps.jsonl").open("x") as stream:
            for step in range(config["online_max_steps"]):
                measured, opening = source_pose(obs), aperture(obs)
                anchor_tcp = measured @ h
                need_query = targets is None
                if source == "wam" and not need_query:
                    need_query = (
                        bool(governed["complete"]) if variant["governor"] else index >= len(targets)
                    ) or step - chunk_step >= config["max_observation_interval_steps"]
                if need_query:
                    before = np.asarray(env.sim.get_state().flatten()).copy()
                    before_time, query_start = float(env.sim.data.time), time.monotonic()
                    if source == "wam":
                        actions, image_sha = generate(
                            c, render_current_cameras(env, c["render_resolution"]), dest, len(chunks)
                        )
                        targets, openings = decode_action_targets(
                            anchor_tcp, actions, horizon=config["online_execution_horizon"]
                        )
                    else:
                        targets, openings = reference @ h, reference_opening
                        actions, image_sha = None, None
                    after = np.asarray(env.sim.get_state().flatten())
                    if float(env.sim.data.time) != before_time or not np.array_equal(before, after):
                        raise RuntimeError("physical state advanced while generating a reference")
                    summary = None
                    if variant["governor"]:
                        summary = governor.set_reference(
                            anchor_tcp, targets, openings, anchor_opening=opening, model_dt=dt
                        )
                    index, chunk_step = 0, step
                    chunks.append(
                        dict(
                            step=step,
                            anchor_tcp=anchor_tcp,
                            targets_tcp=targets,
                            openings=openings,
                            actions=actions,
                            image_sha256=image_sha,
                            physics_sha256=sha256_bytes(before.tobytes()),
                            physics_unchanged=True,
                            generation_s=time.monotonic() - query_start,
                            retiming=summary,
                        )
                    )
                    write_json(dest / "chunks.json", chunks)
                if variant["governor"]:
                    governed = governor.advance(anchor_tcp, dt)
                    desired_tcp, desired_opening = governed["target_tcp"], governed["opening"]
                else:
                    desired_tcp, desired_opening = (
                        targets[min(index, len(targets) - 1)],
                        float(openings[min(index, len(openings) - 1)]),
                    )
                    index += 1
                desired = desired_tcp @ h_inv
                twist, goal_before = measured_twist(env), np.asarray(controller.goal_ori).copy()
                converted = converter.convert(measured, desired, twist)
                arm, rotation_result = converted["command"].copy(), None
                if variant["rotation"] == "bounded":
                    rotation_result = bounded_osc_rotation_inverse(
                        measured[:3, :3],
                        desired[:3, :3],
                        twist[3:],
                        interface=interface,
                        kp=contract["kp"][3:],
                        kd=contract["kd"][3:],
                        dt=dt,
                        previous_command=previous_command[3:6],
                        current_goal_rotation=goal_before,
                        max_command_rate=config["max_command_rate"],
                        **config["bounded_rotation"],
                    )
                    arm[3:] = rotation_result["command"]
                    converter.previous_command = arm.copy()
                motor_opening = panda_opening_from_motor_target(env.robots[0].gripper.current_action)
                grip_result = None
                if variant["gripper"] == "tracking":
                    grip_result = gripper.update(
                        desired_opening,
                        opening,
                        motor_opening=motor_opening,
                        previous_command=previous_command[-1],
                        dt=dt,
                    )
                    grip_command = grip_result.command
                else:
                    grip_command = intent.update(desired_opening)
                command = np.r_[arm, grip_command]
                before_time = float(env.sim.data.time)
                env.step(command)
                obs = current_observation(env)
                after_pose = source_pose(obs)
                progress = object_progress(env)
                for name, obj in progress["objects"].items():
                    obj["height_gain_m"] = float(obj["position"][2] - initial_objects[name]["position"][2])
                row = dict(
                    step=step,
                    chunk_index=len(chunks) - 1,
                    measured_source=measured,
                    measured_twist=twist,
                    desired_source=desired,
                    desired_tcp=desired_tcp,
                    after_source=after_pose,
                    opening_target=desired_opening,
                    opening_before=opening,
                    opening_after=aperture(obs),
                    motor_opening_before=motor_opening,
                    motor_opening_after=panda_opening_from_motor_target(env.robots[0].gripper.current_action),
                    controller_goal_before=goal_before,
                    controller_goal_after=np.asarray(controller.goal_ori).copy(),
                    command=command,
                    actual_dt=float(env.sim.data.time) - before_time,
                    source_error=pose_error(desired, after_pose),
                    tcp_error=pose_error(desired_tcp, after_pose @ h),
                    reference_progress=governed,
                    rotation_inverse=rotation_result,
                    gripper=asdict(grip_result) if grip_result is not None else None,
                    saturated=converted["saturated"],
                    rate_limited=converted["rate_limited"],
                    **progress,
                )
                rows.append(row)
                previous_command = command.copy()
                stream.write(json.dumps(row, default=lambda value: value.tolist(), allow_nan=False) + "\n")
                if step % 20 == 19:
                    stream.flush()
                    status = dict(steps=step + 1, chunks=len(chunks), ever_success=any(r["success"] for r in rows))
                    write_json(dest / "progress.json", status)
                    print(json.dumps(dict(record=record["name"], variant=variant_name, **status)), flush=True)
    except BaseException as exc:
        failure = repr(exc)
    finally:
        env.close()
    if not failure:
        for filename, signature in execution_sha256.items():
            if sha256(filename) != signature:
                failure = f"source changed during trial: {filename}"
                break
    result = dict(
        record=record["name"],
        task=record["task"],
        source=source,
        checkpoint=checkpoint,
        variant=variant_name,
        sampling_seed=sampling_seed,
        steps=len(rows),
        expected_steps=config["online_max_steps"],
        error=failure,
        chunks=len(chunks),
        elapsed_s=time.monotonic() - start,
        inference_latency_excluded=True,
        stable_success=bool(
            len(rows) >= config["stable_success_last_steps"]
            and all(r["success"] for r in rows[-config["stable_success_last_steps"] :])
        ),
        ever_success=any(r["success"] for r in rows),
    )
    if rows:
        errors = np.asarray([r["tcp_error"] for r in rows])
        result.update(
            tcp_position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            opening_mae=float(np.mean([abs(r["opening_after"] - r["opening_target"]) for r in rows])),
            first_success_step=next((r["step"] for r in rows if r["success"]), None),
            final_success_steps=sum(r["success"] for r in rows[-config["stable_success_last_steps"] :]),
            physical_elapsed_s=sum(r["actual_dt"] for r in rows),
        )
    write_json(dest / "result.json", result)
    if failure:
        raise RuntimeError(failure)
    return result
