"""Frozen multi-task physics evaluation and independent executor qualification."""

import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.action_reference import decode_action_targets, encode_oracle_targets
from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.cameras import render_current_cameras
from s2a.geometry import inverse, pose_error
from s2a.gripper import OpeningIntent, opening_position_target
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


def object_progress(env):
    """Native task predicate plus separately labeled grasp/contact progress."""
    core = env.env
    objects = {}
    for name, obj in core.objects_dict.items():
        objects[name] = dict(
            position=env.sim.data.body_xpos[core.obj_body_id[name]].copy(),
            grasp_contact=bool(core._check_grasp(core.robots[0].gripper, obj)),
        )
    return dict(success=bool(env.check_success()), objects=objects)


def run_case(config, root, record, *, source, sampling_seed, checkpoint=None, probe=None, server_port=None):
    if source not in ("native", "oracle", "wam", "probe"):
        raise ValueError("unknown source")
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
        execution_horizon=config["online_execution_horizon"],
        max_steps=config["online_max_steps"],
        sampling_steps=config["sampling_steps"],
        guidance=config["guidance"],
        shift=config["shift"],
        control_fps=config["control_fps"],
        inverse_weight=config["inverse_weight"],
        max_command_rate=config["max_command_rate"],
        opening_hysteresis=config["opening_hysteresis"],
    )
    label = checkpoint or source
    if probe:
        label += "_" + probe["name"]
    dest = root / "simulation" / label / f"{record['name']}_s{sampling_seed}"
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        code=code_fingerprint(),
        config=c,
        record=record,
        source=source,
        probe=probe,
        inference_latency_excluded=True,
        fixed_reference_converter=True,
        runtime_versions={
            p: importlib.metadata.version(p) for p in ("mujoco", "robosuite", "numpy", "scipy", "torch", "h5py")
        },
    )
    if source == "wam":
        import requests

        ck = next(x for x in config["checkpoints"] if x["name"] == checkpoint)
        c["server_url"] = f"http://127.0.0.1:{server_port or ck['port']}"
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(c["server_url"] + "/info", timeout=15)
            response.raise_for_status()
            info = response.json()
        expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
        if Path(info["checkpoint_path"]).resolve() != expected.resolve() or info["ema_weights"]:
            raise ValueError("wrong checkpoint")
        manifest["server_info"] = info
    h = np.asarray(json.loads(Path(c["tool_contract"]).read_text())["source_from_tcp"])
    manifest["tool_sha256"] = sha256(c["tool_contract"])
    native = None
    if source == "native":
        if sha256(record["expert_path"]) != record["expert_sha256"]:
            raise ValueError("changed expert source")
        native = np.load(record["expert_path"])
    if source == "oracle":
        reference_path = root / "simulation/native" / f"{record['name']}_s{sampling_seed}/reference.npz"
        manifest["reference_sha256"] = sha256(reference_path)
        with np.load(reference_path) as data:
            reference, opening_reference = data["poses"], data["openings"]
    write_json(dest / "manifest.json", manifest)
    env, _ = make_environment(c, dest, record, 0)
    rows, chunks, error = [], [], None
    start = time.monotonic()
    count = 180 if source == "probe" else config["online_max_steps"]
    try:
        obs = current_observation(env)
        initial_pose = source_pose(obs)
        initial_opening = aperture(obs)
        initial_objects = object_progress(env)["objects"]
        _, contract = controller_contract(env)
        if not np.isclose(env.env.control_timestep, 1 / config["control_fps"]):
            raise ValueError("wrong control interval")
        converter = OSCStateToAction(
            OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max"))),
            contract["kp"],
            contract["kd"],
            1 / config["control_fps"],
            compensate=True,
            inverse_weight=config["inverse_weight"],
            max_command_rate=config["max_command_rate"],
        )
        intent = OpeningIntent(initial_opening, band=config["opening_hysteresis"])
        if source != "native":
            env.robots[0].gripper.current_action = opening_position_target(initial_opening)
        write_json(dest / "controller.json", contract)
        targets, target_index, openings = None, 0, None
        with (dest / "steps.jsonl").open("w") as stream:
            for step in range(count):
                measured = source_pose(obs)
                if source == "native":
                    command = native[step] if step < len(native) else np.r_[np.zeros(6), native[-1, -1]]
                    desired, desired_opening = measured, aperture(obs)
                    saturated, rate_limited = False, False
                else:
                    if source == "probe":
                        desired = initial_pose.copy()
                        axis = min(step // 60, 2)
                        value = np.sin(2 * np.pi * ((step % 60) + 1) / 60)
                        desired[axis, 3] += probe["position_amplitude_m"] * value
                        angle = np.eye(3)[axis] * probe["rotation_amplitude_rad"] * value
                        desired[:3, :3] = Rotation.from_rotvec(angle).as_matrix() @ initial_pose[:3, :3]
                        desired_opening = initial_opening
                    else:
                        if targets is None or target_index == len(targets):
                            before = np.asarray(env.sim.get_state().flatten()).copy()
                            before_time = float(env.sim.data.time)
                            anchor = measured @ h
                            query_time = time.monotonic()
                            if source == "oracle":
                                indices = np.minimum(step + np.arange(c["action_chunk_size"]), len(reference) - 1)
                                actions = encode_oracle_targets(
                                    anchor, reference[indices] @ h, opening_reference[indices]
                                )
                                image_hash = None
                            else:
                                actions, image_hash = generate(
                                    c, render_current_cameras(env, c["render_resolution"]), dest, len(chunks)
                                )
                            after = np.asarray(env.sim.get_state().flatten())
                            if float(env.sim.data.time) != before_time or not np.array_equal(before, after):
                                raise RuntimeError("physics changed during inference")
                            targets, openings = decode_action_targets(anchor, actions, horizon=c["execution_horizon"])
                            target_index = 0
                            chunk = dict(
                                step=step,
                                anchor_tcp=anchor,
                                targets_tcp=targets,
                                actions=actions,
                                image_sha256=image_hash,
                                physics_sha256=sha256_bytes(before.tobytes()),
                                physics_unchanged=True,
                                generation_s=time.monotonic() - query_time,
                            )
                            chunks.append(chunk)
                            write_json(dest / "chunks.json", chunks)
                        desired = targets[target_index] @ inverse(h)
                        desired_opening = float(openings[target_index])
                        target_index += 1
                    converted = converter.convert(measured, desired, measured_twist(env))
                    command = np.r_[converted["command"], intent.update(desired_opening)]
                    saturated, rate_limited = converted["saturated"], converted["rate_limited"]
                before_time = float(env.sim.data.time)
                env.step(command)
                obs = current_observation(env)
                after_pose = source_pose(obs)
                progress = object_progress(env)
                for name, obj in progress["objects"].items():
                    obj["height_gain_m"] = float(obj["position"][2] - initial_objects[name]["position"][2])
                    obj["displacement_m"] = float(np.linalg.norm(obj["position"] - initial_objects[name]["position"]))
                row = dict(
                    step=step,
                    measured_source=measured,
                    desired_source=desired,
                    after_source=after_pose,
                    opening_target=desired_opening,
                    opening_after=aperture(obs),
                    command=command,
                    actual_dt=float(env.sim.data.time) - before_time,
                    error=pose_error(desired, after_pose),
                    saturated=saturated,
                    rate_limited=rate_limited,
                    **progress,
                )
                rows.append(row)
                stream.write(json.dumps(row, default=lambda x: x.tolist(), allow_nan=False) + "\n")
                stream.flush()
                if step % 8 == 7:
                    status = dict(
                        steps=step + 1,
                        chunks=len(chunks),
                        success=any(r["success"] for r in rows),
                        elapsed_s=time.monotonic() - start,
                    )
                    write_json(dest / "progress.json", status)
                    print(json.dumps(dict(name=record["name"], source=label, **status)), flush=True)
        if source == "native":
            np.savez_compressed(
                dest / "reference.npz",
                poses=np.asarray([r["after_source"] for r in rows]),
                openings=np.asarray([r["opening_after"] for r in rows]),
            )
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
    result = dict(
        record=record["name"],
        source=source,
        checkpoint=checkpoint,
        sampling_seed=sampling_seed,
        steps=len(rows),
        expected_steps=count,
        error=error,
        success=any(r["success"] for r in rows),
        terminal_success=bool(rows and rows[-1]["success"]),
        chunks=len(chunks),
        elapsed_s=time.monotonic() - start,
        physical_elapsed_s=sum(r["actual_dt"] for r in rows),
        inference_latency_excluded=True,
    )
    if rows:
        errors = np.asarray([r["error"] for r in rows])
        result.update(
            position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            opening_mae=float(np.mean([abs(r["opening_after"] - r["opening_target"]) for r in rows])),
            first_success_step=next((r["step"] for r in rows if r["success"]), None),
            final_second_success_fraction=float(np.mean([r["success"] for r in rows[-20:]])),
            object_progress={
                name: dict(
                    grasp_contact_steps=sum(r["objects"][name]["grasp_contact"] for r in rows),
                    max_height_gain_m=max(r["objects"][name]["height_gain_m"] for r in rows),
                    max_displacement_m=max(r["objects"][name]["displacement_m"] for r in rows),
                )
                for name in rows[0]["objects"]
            },
        )
        result["tracking_applicable"] = source != "native"
        if source == "native":
            result["step_motion_position_rms_m"] = result.pop("position_rmse_m")
            result["step_motion_rotation_rms_rad"] = result.pop("rotation_rmse_rad")
            result["opening_step_mae"] = result.pop("opening_mae")
    write_json(dest / "result.json", result)
    if error:
        raise RuntimeError(error)
    return result
