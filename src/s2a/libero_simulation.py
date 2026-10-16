"""Independent LIBERO forward execution, preserving the historical protocol.

LIBERO assets and the original episode fixture are read-only. No Cosmos
training reader is imported. Heavy dependencies are optional and loaded lazily.
"""

import ctypes
import importlib.metadata
import json
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.geometry import inverse, pose_error, rot6d_to_matrix, transform
from s2a.osc import OSCInterface, OSCStateToAction


def configure_simulator(config, output):
    config_dir = output / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_bytes(Path(config["libero_config"]).read_bytes())
    os.environ.update(
        LIBERO_CONFIG_PATH=str(config_dir),
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        NUMBA_CACHE_DIR=str(output / "numba_cache"),
    )
    sys.path.insert(0, config["libero_root"])
    # Resolve only the missing system OSMesa library. Python still starts with
    # LD_LIBRARY_PATH='' as required in this NGC environment.
    ctypes.CDLL(config["osmesa_library"], mode=ctypes.RTLD_GLOBAL)
    import mujoco
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    groups = mujoco.MjvOption().geomgroup
    if groups.strides != (groups.itemsize,):
        raise RuntimeError("MuJoCo array layout incompatible with NumPy")
    return benchmark, get_libero_path, OffScreenRenderEnv


def source_pose(obs):
    return transform(obs["robot0_eef_pos"], Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix())


def make_environment(config, output, record, trial):
    benchmark, get_path, env_class = configure_simulator(config, output)
    task = benchmark.get_benchmark_dict()[config["task_suite"]]().get_task(config["task_id"])
    random.seed(config["seed"] + trial)
    np.random.seed(config["seed"] + trial)
    env = env_class(
        bddl_file_name=str(Path(get_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=config["render_resolution"],
        camera_widths=config["render_resolution"],
        render_gpu_device_id=-1,
    )
    env.seed(config["seed"] + trial)
    env.reset()
    from libero.libero.utils.utils import postprocess_model_xml

    xml = ET.fromstring(postprocess_model_xml(record["model_xml"], {}))
    for node in xml.iter():
        file = node.get("file")
        if file and "/assets/" in file and not Path(file).is_file():
            asset = Path(get_path("assets")) / file.split("/assets/", 1)[1]
            if not asset.is_file():
                raise FileNotFoundError(asset)
            node.set("file", str(asset))
    env.reset_from_xml_string(ET.tostring(xml, encoding="unicode"))
    initial = np.asarray(record["initial_state"], dtype=float)
    if sha256_bytes(initial.tobytes()) != record["initial_state_sha256"]:
        env.close()
        raise ValueError("fixture initial state hash mismatch")
    obs = env.set_init_state(initial)
    return env, obs


def sha256_bytes(data):
    import hashlib

    return hashlib.sha256(data).hexdigest()


def canvas(images):
    """Exact historical training cover-crop, torch bilinear, 4px seam."""
    import torch

    result = torch.empty((3, 1, 960, 640), dtype=torch.uint8)
    for image, start in zip(images, (0, 480), strict=True):
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float()
        scale = max(640 / pixels.shape[3], 480 / pixels.shape[2])
        height, width = math.ceil(pixels.shape[2] * scale), math.ceil(pixels.shape[3] * scale)
        resized = torch.nn.functional.interpolate(pixels, (height, width), mode="bilinear", align_corners=False)
        y, x = (height - 480) // 2, (width - 640) // 2
        result[:, 0, start : start + 480] = resized[0, :, y : y + 480, x : x + 640].round().clamp(0, 255)
    result[:, :, 478:482] = 0
    return result[:, 0].permute(1, 2, 0).numpy()


def generate(config, obs, dest, number):
    import requests
    from PIL import Image

    images = [obs[k][::-1, ::-1].copy() for k in ("agentview_image", "robot0_eye_in_hand_image")]
    path = dest / f"input_{number:04d}.png"
    pixels = canvas(images)
    Image.fromarray(pixels).save(path)
    request = dict(
        name=f"s2a_{dest.name}_{number}",
        model_mode="wam",
        prompt=config["task_text"],
        vision_path=str(path),
        domain_name="franka_panda__parallel_gripper",
        image_size=960,
        action_chunk_size=config["action_chunk_size"],
        fps=config["control_fps"],
        num_frames=config["action_chunk_size"] + 1,
        seed=config["seed"],
        num_steps=config["sampling_steps"],
        guidance=config["guidance"],
        shift=config["shift"],
        extra={
            "action128": dict(
                action_domain=config["action_domain"],
                action_dim_mask=[True] * 10 + [False] * 118,
                action_fps=config["control_fps"],
                task_title=config["task_text"],
                selected_roles=["auxiliary_0", "left_wrist"],
                camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
            )
        },
    )
    write_json(dest / f"request_{number:04d}.json", request)
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(config["server_url"] + "/generate", json=request, timeout=900)
        response.raise_for_status()
        result = response.json()
    write_json(dest / f"response_{number:04d}.json", result)
    if result["status"] != "success":
        raise RuntimeError(result.get("message", result))
    action = np.asarray(result["outputs"][0]["content"]["action"], dtype=float)
    if action.shape != (config["action_chunk_size"], 128) or not np.isfinite(action).all() or action[:, 10:].any():
        raise ValueError("invalid model Action128 output")
    return action, sha256_bytes(pixels.tobytes())


def expert_reference(fixture, record, h):
    import h5py

    with h5py.File(fixture["source_hdf5"]) as source:
        demo = source["data"][record["demo"]]
        native = demo["actions"][:]
        rotations = Rotation.from_rotvec(demo["obs/ee_ori"][:]).as_matrix()
        poses = transform(demo["obs/ee_pos"][:], rotations) @ h
        grip = demo["obs/gripper_states"][:]
    deltas = inverse(poses[:-1]) @ poses[1:]
    aperture = np.clip(grip[:, 0] - grip[:, 1] - 0.001, 0, 0.079) / 0.079
    actions = np.column_stack((deltas[:, :3, 3], deltas[:, :3, 0], deltas[:, :3, 1], aperture[1:]))
    return actions, poses, aperture, native


def controller_contract(env):
    controller = env.robots[0].controller
    if not controller.use_delta or controller.impedance_mode != "fixed":
        raise ValueError("requires the historical fixed delta OSC_POSE controller")
    if controller.interpolator_pos is not None or controller.interpolator_ori is not None:
        raise ValueError("inner response assumes a held goal, not an interpolator")
    info = {
        k: np.asarray(getattr(controller, k))
        for k in ("input_min", "input_max", "output_min", "output_max", "kp", "kd")
    }
    info.update(control_freq=env.env.control_freq, model_timestep=env.env.model_timestep)
    return controller, info


def measured_twist(env):
    controller = env.robots[0].controller
    return np.r_[env.sim.data.get_site_xvelp(controller.eef_name), env.sim.data.get_site_xvelr(controller.eef_name)]


def aperture(obs):
    q = np.asarray(obs["robot0_gripper_qpos"])
    return float(np.clip((q[0] - q[1] - 0.001) / 0.079, 0, 1))


def run_trial(config, output, *, stage, candidate, trial, max_steps=None):
    """Execute actual MuJoCo steps; save pre/post state, target and sent action."""
    import imageio.v2 as imageio

    fixture = json.loads(Path(config["fixture"]).read_text())
    record = fixture["episodes"][trial]
    if record["episode"] != config["episodes"][trial]:
        raise ValueError("frozen episode selection mismatch")
    h = np.asarray(json.loads(Path(config["tool_contract"]).read_text())["source_from_tcp"])
    dest = output / stage / candidate / f"trial{trial}"
    dest.mkdir(parents=True, exist_ok=False)
    provenance = {
        "config": config,
        "code": code_fingerprint(),
        "inputs_sha256": {p: sha256(p) for p in (config["fixture"], config["tool_contract"], fixture["source_hdf5"])},
        "runtime": {
            p: importlib.metadata.version(p) for p in ("mujoco", "robosuite", "numpy", "scipy", "torch", "h5py")
        },
    }
    if stage == "policy":
        import requests

        with requests.Session() as session:
            session.trust_env = False
            response = session.get(config["server_url"] + "/info", timeout=15)
            response.raise_for_status()
        server = response.json()
        expected = Path(config["checkpoint_run"]) / "checkpoints" / config["checkpoint_iteration"] / "model"
        if Path(server["checkpoint_path"]).resolve() != expected.resolve():
            raise ValueError("server loaded a different checkpoint")
        provenance["server_info"] = server
    write_json(dest / "manifest.json", provenance)
    env, obs = make_environment(config, output, record, trial)
    started = time.monotonic()
    rows, requests_log = [], []
    success, error = False, None
    writer = None
    try:
        controller, contract = controller_contract(env)
        dt = 1 / config["control_fps"]
        if not np.isclose(env.env.control_timestep, dt):
            raise ValueError("simulation control clock differs from frozen protocol")
        converter = OSCStateToAction(
            OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max"))),
            contract["kp"],
            contract["kd"],
            dt,
            compensate=candidate == "compensated",
            inverse_weight=config["inverse_weight"],
        )
        write_json(
            dest / "controller.json", {**contract, "response": converter.response, "momentum": converter.momentum}
        )
        reference, absolute_poses, openings, native = expert_reference(fixture, record, h)
        length = config["max_steps"] if max_steps is None else max_steps
        if stage in ("oracle", "native"):
            length = len(reference) + (config["oracle_extra_hold_steps"] if stage == "oracle" else 0)
        if stage == "probe":
            length = config["probe_steps"]
        generator = np.random.default_rng(config["seed"] + 1000)
        anchor = source_pose(obs) @ h
        gripper_target = np.zeros(2)
        gripper = env.robots[0].gripper

        def position_action(_action):
            gripper.current_action = gripper_target.copy()
            return gripper.current_action

        if stage != "native":
            gripper.format_action = position_action
        writer = imageio.get_writer(dest / "rollout.mp4", fps=config["control_fps"])
        queue = []
        with (dest / "trace_steps.jsonl").open("w") as stream:
            for step in range(length):
                current = source_pose(obs)
                twist = measured_twist(env)
                writer.append_data(obs["agentview_image"][::-1, ::-1])
                if stage == "policy" and not queue:
                    actions, observation_hash = generate(config, obs, dest, len(requests_log))
                    queue = list(actions[: config["execution_horizon"]])
                    anchor = current @ h
                    requests_log.append(dict(step=step, observation_sha256=observation_hash))
                if stage == "policy":
                    action = queue.pop(0)
                    anchor = anchor @ transform(action[:3], rot6d_to_matrix(action[3:9]))
                    desired_tcp, opening = anchor.copy(), float(np.clip(action[9], 0, 1))
                elif stage == "probe":
                    motion = generator.uniform(-1, 1, 6) * np.array(
                        [config["probe_translation_amplitude_m"]] * 3 + [config["probe_rotation_amplitude_rad"]] * 3
                    )
                    desired = transform(
                        current[:3, 3] + motion[:3], Rotation.from_rotvec(motion[3:]).as_matrix() @ current[:3, :3]
                    )
                    desired_tcp, opening = desired @ h, 1.0
                else:
                    index = min(step + 1, len(absolute_poses) - 1)
                    desired_tcp, opening = absolute_poses[index], float(openings[index])
                desired = desired_tcp @ inverse(h)
                result = converter.convert(current, desired, twist)
                desired_q = (0.079 * opening + 0.001) / 2
                internal = 2 * desired_q / 0.04 - 1
                gripper_target[:] = [internal, -internal]
                executed = np.r_[result["command"], 0.0] if stage != "native" else native[step]
                before_time = float(env.sim.data.time)
                next_obs, _, done, info = env.step(executed)
                actual_dt = float(env.sim.data.time) - before_time
                after = source_pose(next_obs)
                row = {
                    "step": step,
                    "simulation_time": before_time,
                    "actual_dt": actual_dt,
                    "measured_source": current.tolist(),
                    "desired_source": desired.tolist(),
                    "desired_tcp": desired_tcp.tolist(),
                    "after_source": after.tolist(),
                    "tracking_error_source": pose_error(desired, after).tolist(),
                    "tracking_error_tcp": pose_error(desired_tcp, after @ h).tolist(),
                    "actual_motion": pose_error(after, current).tolist(),
                    "opening_target": opening,
                    "opening_after": aperture(next_obs),
                    "gripper_actuator_target": gripper_target.tolist() if stage != "native" else None,
                    "executed": executed.tolist(),
                    "converter": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()},
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                obs = next_obs
                success = bool(info.get("success", done))
                if step % 8 == 0 or success:
                    progress = dict(
                        steps=step + 1,
                        success=success,
                        requests=len(requests_log),
                        elapsed_s=time.monotonic() - started,
                    )
                    write_json(dest / "progress.json", progress)
                if step % 80 == 0 or success:
                    print(json.dumps(dict(stage=stage, candidate=candidate, trial=trial, **progress)), flush=True)
                if success:
                    break
    except Exception as exc:
        error = repr(exc)
    finally:
        if writer is not None:
            writer.close()
        env.close()
    result = dict(
        stage=stage,
        candidate=candidate,
        trial=trial,
        episode=record["episode"],
        seed=config["seed"] + trial,
        initial_state_sha256=record["initial_state_sha256"],
        success=success,
        error=error,
        steps=len(rows),
        elapsed_s=time.monotonic() - started,
        requests=requests_log,
    )
    if rows:
        errors = np.array([r["tracking_error_tcp"] for r in rows])
        command = np.array([r["executed"][:6] for r in rows])
        result.update(
            position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            aperture_rmse=float(np.sqrt(np.mean([(r["opening_target"] - r["opening_after"]) ** 2 for r in rows]))),
            saturation_fraction=float(np.mean([r["converter"]["saturated"] for r in rows])),
            max_command_change=float(np.max(np.linalg.norm(np.diff(command, axis=0), axis=1))) if len(rows) > 1 else 0,
            actual_dt_range_s=[min(r["actual_dt"] for r in rows), max(r["actual_dt"] for r in rows)],
        )
        np.savez_compressed(dest / "trace.npz", errors=errors, command=command)
    write_json(dest / "result.json", result)
    print(json.dumps(result), flush=True)
    if error:
        raise RuntimeError(error)
    return result
