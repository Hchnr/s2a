#!/usr/bin/env python3
"""Collect one fixed task of synchronous native trajectories on CPU/OSMesa."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from s2a.artifacts import sha256  # noqa: E402
from s2a.geometry import transform, validate_transform  # noqa: E402
from s2a.native_training import (  # noqa: E402
    native_anchor_indices,
    native_command_schedule,
    state_sha256,
    synchronous_labels,
    validate_physical_source,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cpu_environment():
    expected = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    actual = {key: os.environ.get(key) for key in expected}
    require(actual == expected, f"CPU environment required: {actual}")
    return actual


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def write_once(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert) + "\n"
    with Path(path).open("x") as stream:
        stream.write(encoded)
    return reference(path)


def save_arrays(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return reference(path)


def raw_state(env):
    state = np.asarray(env.sim.get_state().flatten()).copy()
    state_sha256(state)
    return state


def endpoint(env, index):
    from s2a.libero_simulation import aperture, current_observation, source_pose

    obs = current_observation(env)
    robot = env.robots[0]
    controller = robot.controller
    sim = env.sim
    source_site_id = robot.eef_site_id
    body_name = robot.robot_model.eef_name
    source_site_pose = transform(
        np.asarray(sim.data.site_xpos[source_site_id]).copy(),
        np.asarray(sim.data.site_xmat[source_site_id]).reshape(3, 3).copy(),
    )
    body_pose = transform(
        np.asarray(sim.data.get_body_xpos(body_name)).copy(),
        np.asarray(sim.data.get_body_xmat(body_name)).reshape(3, 3).copy(),
    )
    site_pose = transform(
        np.asarray(sim.data.get_site_xpos(controller.eef_name)).copy(),
        np.asarray(sim.data.get_site_xmat(controller.eef_name)).reshape(3, 3).copy(),
    )
    source = source_pose(obs)
    physical = validate_physical_source(source, source_site_pose, body_pose)
    state = raw_state(env)
    sample = dict(
        endpoint_step=np.int64(index),
        sim_time=np.float64(sim.data.time),
        native_success=np.bool_(env.check_success()),
        sim_states=state,
        source_position=np.asarray(obs["robot0_eef_pos"]).copy(),
        source_quaternion_xyzw=np.asarray(obs["robot0_eef_quat"]).copy(),
        gripper_qpos=np.asarray(obs["robot0_gripper_qpos"]).copy(),
        source_pose=source,
        actual_source_site_pose=source_site_pose,
        actual_eef_body_pose=body_pose,
        physical_source_pose=physical,
        controller_site_pose=validate_transform(site_pose),
        opening=np.float64(aperture(obs)),
        gripper_current_action=np.asarray(robot.gripper.current_action).copy(),
    )
    # Preserve the controller's actual cached state. Do not change its goals or
    # force a new update between native commands.
    for key in ("ee_pos", "ee_ori_mat", "goal_pos", "goal_ori", "joint_pos", "joint_vel"):
        sample[f"controller_{key}"] = np.asarray(getattr(controller, key)).copy()
    for key, value in sample.items():
        require(np.isfinite(value).all(), f"non-finite endpoint {key}")
    receipt = dict(
        endpoint_step=index,
        sim_time=float(sample["sim_time"]),
        sim_state_sha256=state_sha256(state),
        sim_state_dtype=str(state.dtype),
        sim_state_shape=list(state.shape),
        native_success=bool(sample["native_success"]),
    )
    return sample, receipt


def render_anchor(env, frame, current, receipt, case_dir):
    from PIL import Image

    from s2a.cameras import render_current_cameras
    from s2a.libero_simulation import canvas

    destination = case_dir / f"anchor{frame:04d}"
    destination.mkdir()
    before = raw_state(env)
    before_time = float(env.sim.data.time)
    require(state_sha256(before) == receipt["sim_state_sha256"], "render not at recorded endpoint")
    frames = render_current_cameras(env, 256)
    expected_keys = ("agentview_image", "robot0_eye_in_hand_image")
    require(set(frames) == set(expected_keys), "unexpected camera roles")
    for pixels in frames.values():
        require(pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8, "invalid live RGB")
    images = [frames[key][::-1, ::-1].copy() for key in expected_keys]
    pixels = canvas(images)
    require(pixels.shape == (960, 640, 3) and pixels.dtype == np.uint8, "invalid live canvas")
    after = raw_state(env)
    after_time = float(env.sim.data.time)
    require(np.array_equal(before, after) and before_time == after_time, "render advanced/restored state")
    camera_arrays = {key: frames[key] for key in expected_keys}
    camera_arrays.update(live_auxiliary_0=images[0], live_left_wrist=images[1])
    camera_ref = save_arrays(destination / "current_cameras.npz", camera_arrays)
    png = destination / "current.png"
    with png.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG")
    binding = dict(sim_time=before_time, sim_state_sha256=state_sha256(before))
    return dict(
        frame_index=frame,
        current_endpoint=receipt,
        current_source_pose=current["source_pose"],
        current_opening=float(current["opening"]),
        render_before=binding,
        render_after=dict(sim_time=after_time, sim_state_sha256=state_sha256(after)),
        raw_cameras=camera_ref,
        camera_keys=list(expected_keys),
        camera_roles=["auxiliary_0", "left_wrist"],
        raw_camera_shape=[256, 256, 3],
        raw_camera_dtype="uint8",
        live_flip_axes=[0, 1],
        canvas_png=reference(png),
        canvas_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        canvas_shape=list(pixels.shape),
        canvas_dtype=str(pixels.dtype),
        future_rgb_frames=0,
    )


def collect_case(prepared, prepared_ref, record, case_dir, progress):
    from s2a.libero_simulation import controller_contract, make_environment

    require(sha256(record["expert_path"]) == record["expert_sha256"], "selected native command hash changed")
    native = np.load(record["expert_path"], allow_pickle=False)
    commands = native_command_schedule(native)
    require(len(native) == record["native_steps"], "native length differs from frozen roster")
    anchors = native_anchor_indices(len(native))
    require(anchors == record["anchor_frames"], "anchor roster differs")
    h = validate_transform(prepared["tool_source_from_tcp"])
    policy = dict(prepared["policy_config"])
    policy.update(
        task_id=record["task_id"],
        task_text=record["task"],
        seed=42,
        control_fps=20,
        render_resolution=256,
        camera_observations=False,
        refresh_controller_after_restore=True,
    )
    case_dir.mkdir()
    env = None
    samples, receipts, rendered, sent = [], [], [], []
    try:
        progress["environment_initializations"] += 1
        env, _ = make_environment(policy, case_dir, record, trial=0)
        controller, contract = controller_contract(env)
        require(contract["control_freq"] == 20, "native control frequency changed")
        require(np.isclose(env.env.control_timestep, 0.05, atol=1e-12, rtol=0), "native control dt changed")
        require(env.env.action_dim == 7, "native action dimension changed")
        robot = env.robots[0]
        observation_contract = dict(
            source_position="robot0_eef_pos = actual grip-site position",
            source_rotation="robot0_eef_quat xyzw = actual robot EEF BODY orientation",
            actual_source_site_id=int(robot.eef_site_id),
            actual_source_site_name=env.sim.model.site_id2name(robot.eef_site_id),
            actual_eef_body_name=robot.robot_model.eef_name,
            actual_eef_body_id=int(env.sim.model.body_name2id(robot.robot_model.eef_name)),
            controller_site_name=controller.eef_name,
            controller_state="cached actual fields; no forced endpoint updates",
            source_from_tcp=h,
            tool_contract_ref=prepared["tool_contract_ref"],
            delta_convention="inverse(T_tcp[k]) @ T_tcp[k+1], local right multiply",
            units=dict(position="m", time="s", rotation="column rot6d"),
            labels_normalized=False,
            initial_seed=42,
            initial_trial=0,
            pre_rollout_steps=0,
            initial_controller_refreshed=True,
            camera_observables=False,
        )
        for index in range(len(commands) + 1):
            if index:
                actual = commands[index - 1].copy()
                progress["physics_steps"] += 1
                env.step(actual)
                # Record after the call too: the actual argument must not have
                # been modified by the environment's control implementation.
                require(np.array_equal(actual, commands[index - 1]), "environment mutated sent command")
                sent.append(actual)
            sample, receipt = endpoint(env, index)
            if index == 0:
                require(receipt["sim_state_sha256"] == record["initial_state_sha256"], "restored initial changed")
            samples.append(sample)
            receipts.append(receipt)
            with (case_dir / "endpoints.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt, allow_nan=False) + "\n")
                stream.flush()
            if index in anchors:
                rendered.append(render_anchor(env, index, sample, receipt, case_dir))
                progress["current_canvas_images"] += 1
        arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
        arrays["native_commands"] = native.copy()
        arrays["sent_commands"] = np.stack(sent)
        arrays.update(synchronous_labels(arrays["source_pose"], arrays["opening"], h, arrays["sim_time"]))
        require(len(samples) == len(native) + 41, "endpoint budget mismatch")
        stable = bool(arrays["native_success"][-40:].all())
        arrays_ref = save_arrays(case_dir / "trajectory.npz", arrays)
        case_result = dict(
            schema_version="s2a.native-training-case.v1",
            complete=True,
            passed=True,
            native_quality_passed=stable,
            prepared_ref=prepared_ref,
            record=record,
            native_steps=len(native),
            physics_steps=len(sent),
            endpoint_count=len(samples),
            environment_initializations=1,
            current_canvas_images=len(rendered),
            future_rgb_frames=0,
            arrays=arrays_ref,
            array_index={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()},
            original_dtypes={key: str(value.dtype) for key, value in arrays.items()},
            endpoint_receipts=reference(case_dir / "endpoints.jsonl"),
            controller_contract=contract,
            observation_contract=observation_contract,
            anchors=rendered,
            stable_success_last_steps=40,
            source_hdf5_reads=0,
            native_command_ref=dict(path=record["expert_path"], sha256=record["expert_sha256"]),
        )
        return write_once(case_dir / "result.json", case_result)
    finally:
        if env is not None:
            env.close()


def run(args):
    environment = cpu_environment()
    require(args.seed == 42 and args.task_index in (0, 1, 2), "fixed seed/task required")
    from prepare_wam_native_training import read_prepared

    prepared_path = Path(args.prepared).resolve()
    prepared = read_prepared(prepared_path, args.prepared_sha256)
    require(prepared["schema_version"] == "s2a.native-training-preparation.v1", "preparation schema mismatch")
    require(prepared["seed"] == 42, "prepared seed changed")
    selected = [record for record in prepared["records"] if record["task_index"] == args.task_index]
    require(len(selected) == 10 and len({record["name"] for record in selected}) == 10, "fixed ten cases required")
    destination = Path(prepared["output_root"]) / "collection" / f"task{args.task_index}"
    destination.mkdir(parents=True, exist_ok=False)
    prepared_ref = reference(prepared_path)
    write_once(
        destination / "worker_started.json",
        dict(
            pid=os.getpid(),
            ppid=os.getppid(),
            argv=sys.argv,
            cuda=environment["CUDA_VISIBLE_DEVICES"],
            ld=environment["LD_LIBRARY_PATH"],
            actual_environment=environment,
            task_index=args.task_index,
        ),
    )
    progress = dict(environment_initializations=0, physics_steps=0, current_canvas_images=0)
    result = dict(
        schema_version="s2a.native-training-task.v1",
        prepared_ref=prepared_ref,
        task_index=args.task_index,
        seed=42,
        complete=False,
        passed=False,
        cases=[],
        native_quality_passed=False,
        actual_environment=environment,
        wam_initializations=0,
        denoise_calls=0,
        fits=0,
        fresh_cases=0,
        future_rgb_frames=0,
    )
    try:
        for record in selected:
            result["cases"].append(collect_case(prepared, prepared_ref, record, destination / record["name"], progress))
        read_prepared(prepared_path, args.prepared_sha256)
        qualities = [json.loads(Path(ref["path"]).read_text())["native_quality_passed"] for ref in result["cases"]]
        require(progress["environment_initializations"] == 10, "environment budget mismatch")
        require(progress["current_canvas_images"] == 50, "current image budget mismatch")
        require(progress["physics_steps"] == sum(record["native_steps"] + 40 for record in selected), "step mismatch")
        require(progress["physics_steps"] <= 6400, "task physics budget exceeded")
        result.update(complete=True, passed=True, native_quality_passed=all(qualities))
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    result.update(progress)
    write_once(destination / "result.json", result)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--task-index", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=42)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
