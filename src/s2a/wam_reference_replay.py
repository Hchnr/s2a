"""Independent physics replay of fixed WAM references; no model query or replanning."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.cameras import render_current_cameras
from s2a.checkpoint_simulation import object_progress
from s2a.geometry import pose_error
from s2a.gripper import OpeningIntent, OpeningLatch, opening_position_target
from s2a.libero_simulation import (
    aperture,
    canvas,
    controller_contract,
    current_observation,
    make_environment,
    measured_twist,
    source_pose,
)
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.task_acceptance import task_outcome


def run_replay(root, name, mode):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("reference replay requires explicitly disabled CUDA")
    prepared = json.loads((root / "manifest.json").read_text())
    case = next(x for x in prepared["cases"] if x["name"] == name)
    for source, digest in case["inputs_sha256"].items():
        if sha256(source) != digest:
            raise ValueError("replay source evidence changed")
    c = case["simulation_config"]
    protocol = prepared["config"]
    if mode not in protocol["gripper_modes"]:
        raise ValueError("unregistered replay gripper mode")
    dest = root / "physics" / name / mode
    dest.mkdir(parents=True, exist_ok=False)
    source_path = Path(case["source_result"])
    original = [json.loads(line) for line in source_path.with_name("steps.jsonl").read_text().splitlines()]
    h = np.asarray(json.loads(Path(c["tool_contract"]).read_text())["source_from_tcp"])
    write_json(dest / "manifest.json", dict(code=code_fingerprint(), source=case, config=protocol, mode=mode))
    env, _ = make_environment(c, dest, case["record"], 0)
    rows, render_hashes, error = [], [], None
    try:
        obs = current_observation(env)
        initial_opening = aperture(obs)
        initial_objects = object_progress(env)["objects"]
        _, contract = controller_contract(env)
        expected_contract = json.loads(source_path.with_name("controller.json").read_text())
        if set(contract) != set(expected_contract):
            raise ValueError("replay controller contract differs")
        for key, value in contract.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, expected_contract[key])
            elif value != expected_contract[key]:
                raise ValueError(f"replay controller contract differs: {key}")
        if not np.isclose(env.env.control_timestep, 1 / c["control_fps"]):
            raise ValueError("replay control period differs")
        converter = OSCStateToAction(
            OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max"))),
            contract["kp"],
            contract["kd"],
            1 / c["control_fps"],
            compensate=True,
            inverse_weight=c["inverse_weight"],
            max_command_rate=c["max_command_rate"],
        )
        intent = OpeningIntent(initial_opening, band=c["opening_hysteresis"])
        latch = OpeningLatch(**protocol["opening_latch"])
        gripper = env.robots[0].gripper
        gripper.current_action = opening_position_target(initial_opening)
        position_target = gripper.current_action.copy()
        if mode == "position":

            def position_action(_):
                gripper.current_action = position_target.copy()
                return gripper.current_action

            gripper.format_action = position_action
        write_json(dest / "controller.json", contract)
        with (dest / "steps.jsonl").open("w") as stream:
            for step, reference in enumerate(original):
                # Preserve the source render schedule as well as its physics/feedback schedule.
                if step % c["execution_horizon"] == 0:
                    images = render_current_cameras(env, c["render_resolution"])
                    render_hashes.append(
                        {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in images.items()}
                    )
                    if step == 0:
                        pixels = canvas(
                            [images[key][::-1, ::-1].copy() for key in ("agentview_image", "robot0_eye_in_hand_image")]
                        )
                        Image.fromarray(pixels).save(dest / "initial_input.png")
                        if mode == "intent" and not np.array_equal(
                            pixels, np.asarray(Image.open(source_path.with_name("input_0000.png")))
                        ):
                            raise ValueError("replay initial camera pixels differ")
                measured = source_pose(obs)
                desired = np.asarray(reference["desired_source"])
                opening = float(reference["opening_target"])
                converted = converter.convert(measured, desired, measured_twist(env))
                if mode == "intent":
                    grip = intent.update(opening)
                elif mode == "latch":
                    grip = latch.update(opening)
                else:
                    position_target[:] = opening_position_target(opening)
                    grip = 0.0
                command = np.r_[converted["command"], grip]
                before = float(env.sim.data.time)
                env.step(command)
                obs = current_observation(env)
                after = source_pose(obs)
                progress = object_progress(env)
                for key, obj in progress["objects"].items():
                    obj["height_gain_m"] = float(obj["position"][2] - initial_objects[key]["position"][2])
                row = dict(
                    step=step,
                    command_start_time_s=step / c["control_fps"],
                    endpoint_time_s=(step + 1) / c["control_fps"],
                    measured_source=measured,
                    desired_source=desired,
                    after_source=after,
                    opening_target=opening,
                    opening_after=aperture(obs),
                    command=command,
                    measured_velocity=converted["measured_velocity"],
                    rotation_inverse_enabled=converted["rotation_inverse_enabled"],
                    position_target=position_target.copy() if mode == "position" else None,
                    actual_dt=float(env.sim.data.time) - before,
                    saturated=converted["saturated"],
                    rate_limited=converted["rate_limited"],
                    **progress,
                )
                rows.append(row)
                stream.write(json.dumps(row, default=lambda x: x.tolist(), allow_nan=False) + "\n")
                stream.flush()
        final_images = render_current_cameras(env, c["render_resolution"])
        Image.fromarray(
            canvas([final_images[key][::-1, ::-1].copy() for key in ("agentview_image", "robot0_eye_in_hand_image")])
        ).save(dest / "final_input.png")
        if mode == "intent":
            for key in ("measured_source", "after_source", "command", "opening_after"):
                np.testing.assert_allclose(
                    [x[key] for x in rows], [x[key] for x in original], atol=protocol["replay_atol"], rtol=0
                )
            if [x["success"] for x in rows] != [x["success"] for x in original]:
                raise ValueError("intent replay does not reproduce native task predicates")
            for key in original[0]["objects"]:
                np.testing.assert_allclose(
                    [x["objects"][key]["position"] for x in rows],
                    [x["objects"][key]["position"] for x in original],
                    atol=protocol["replay_atol"],
                    rtol=0,
                )
                if [x["objects"][key]["grasp_contact"] for x in rows] != [
                    x["objects"][key]["grasp_contact"] for x in original
                ]:
                    raise ValueError("intent replay does not reproduce grasp contacts")
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
    result = dict(name=name, mode=mode, steps=len(rows), expected_steps=len(original), error=error)
    if len(rows) == len(original):
        np.testing.assert_allclose([x["actual_dt"] for x in rows], 1 / c["control_fps"], atol=1e-10, rtol=0)
        errors = pose_error(
            np.asarray([x["desired_source"] for x in rows]) @ h,
            np.asarray([x["after_source"] for x in rows]) @ h,
        )
        result.update(
            outcome=task_outcome(
                [x["success"] for x in rows], expected_steps=protocol["steps"], stable_steps=protocol["stable_steps"]
            ),
            position_rmse_tcp_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            original_trajectory_reproduced=mode == "intent" and error is None,
            object_progress={
                key: dict(
                    grasp_contact_steps=sum(x["objects"][key]["grasp_contact"] for x in rows),
                    maximum_height_gain_m=max(x["objects"][key]["height_gain_m"] for x in rows),
                )
                for key in initial_objects
            },
        )
    write_json(dest / "render_hashes.json", render_hashes)
    write_json(dest / "result.json", result)
    if error:
        raise RuntimeError(error)
    return result
