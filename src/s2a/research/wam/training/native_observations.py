"""Shared native endpoint/render capture; no collector or simulator launch."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.geometry import transform, validate_transform
from s2a.infrastructure.artifacts import reference
from s2a.research.wam.training.native.native_training import (
    gripper_cache_receipt,
    state_sha256,
    validate_physical_source,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_arrays(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return reference(path)


def raw_state(env):
    state = np.asarray(env.sim.get_state().flatten()).copy()
    state_sha256(state)
    return state


def endpoint(env, index):
    from s2a.backends.libero.simulation import aperture, current_observation, source_pose

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
        **gripper_cache_receipt(sample["gripper_current_action"]),
    )
    return sample, receipt


def render_anchor(env, frame, current, receipt, case_dir):
    from PIL import Image

    from s2a.backends.libero.cameras import render_current_cameras
    from s2a.backends.libero.simulation import canvas

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
