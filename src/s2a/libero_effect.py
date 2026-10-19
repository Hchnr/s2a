"""Delay-free WAM task evaluation through the same decoder as oracle controls.

World-model inference blocks physics. No stale-reference rejection, time warp,
future physical-state restore, expert control command or fitted frame is used.
"""

import json
import time
from pathlib import Path

import numpy as np

from s2a.action_reference import decode_action_targets, encode_oracle_targets
from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.cameras import render_current_cameras
from s2a.geometry import inverse, pose_error, transform
from s2a.gripper import OpeningIntent, opening_position_target
from s2a.libero_audit import contact_counts
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


def run_effect_case(config, output, case, trial):
    if case["source"] not in ("oracle", "wam") or case["arm"] not in ("affine", "response"):
        raise ValueError("unknown effect comparison")
    if not config["pause_physics_during_inference"] or config["expire_model_references"]:
        raise ValueError("effect evaluation must exclude inference latency")
    config = {**config, "server_url": config["server_urls"][trial]}
    dest = output / "sync" / case["name"] / f"trial{trial}"
    dest.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(Path(config["fixture"]).read_text())
    record = fixture["episodes"][trial]
    h = np.asarray(json.loads(Path(config["tool_contract"]).read_text())["source_from_tcp"])
    inputs = {str(p): sha256(p) for p in (config["fixture"], config["tool_contract"])}
    manifest = dict(config=config, case=case, trial=trial, code=code_fingerprint(), inputs_sha256=inputs)
    if case["source"] == "oracle":
        path = config["published_bridge"]
        reference = json.loads(Path(path).read_text())["trajectories"][record["demo"]]
        published_tcp = np.asarray(reference["source_poses"]) @ h
        published_opening = np.asarray(reference["openings"])
        inputs[path] = sha256(path)
        manifest["evidence"] = "Privileged published endpoints encoded into Action128 and decoded by the WAM path"
        length = config["oracle_steps"]
    else:
        import requests

        with requests.Session() as session:
            session.trust_env = False
            response = session.get(config["server_url"] + "/info", timeout=15)
            response.raise_for_status()
            server = response.json()
        expected = Path(config["checkpoint_run"]) / "checkpoints" / config["checkpoint_iteration"] / "model"
        if Path(server["checkpoint_path"]).resolve() != expected.resolve():
            raise ValueError("server checkpoint differs from the frozen configuration")
        manifest["server_info"] = server
        manifest["evidence"] = (
            "Live-image WAM; inference wall time excluded by paused physics; actual predicted targets executed"
        )
        length = config["max_steps"]
    write_json(dest / "manifest.json", manifest)
    env, obs = make_environment({**config, "camera_observations": False}, dest, record, trial)
    rows, chunks, error = [], [], None
    started = time.monotonic()
    try:
        obs = current_observation(env)
        _, contract = controller_contract(env)
        dt = 1 / config["control_fps"]
        if not np.isclose(env.env.control_timestep, dt):
            raise ValueError("physical control interval differs")
        converter = OSCStateToAction(
            OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max"))),
            contract["kp"],
            contract["kd"],
            dt,
            compensate=case["arm"] == "response",
            inverse_weight=config["inverse_weight"],
            max_command_rate=config["max_command_rate"],
        )
        gripper = env.robots[0].gripper
        gripper.current_action = opening_position_target(aperture(obs))
        intent = OpeningIntent(aperture(obs), band=config["opening_hysteresis"])
        write_json(
            dest / "controller.json",
            dict(**contract, initial_gripper=gripper.current_action.copy(), initial_source=source_pose(obs)),
        )
        initial_time = float(env.sim.data.time)
        target_index, targets, openings, actions = 0, None, None, None
        with (dest / "steps.jsonl").open("w") as stream:
            for step in range(length):
                measured = source_pose(obs)
                if targets is None or target_index == len(targets):
                    anchor = measured @ h
                    before = np.asarray(env.sim.get_state().flatten()).copy()
                    before_time = float(env.sim.data.time)
                    query_started = time.monotonic()
                    number = len(chunks)
                    if case["source"] == "wam":
                        snapshot = render_current_cameras(env, config["render_resolution"])
                        actions, observation_hash = generate(config, snapshot, dest, number)
                    else:
                        indices = np.minimum(step + np.arange(config["action_chunk_size"]), len(published_tcp) - 1)
                        actions = encode_oracle_targets(anchor, published_tcp[indices], published_opening[indices])
                        observation_hash = None
                    after = np.asarray(env.sim.get_state().flatten()).copy()
                    if float(env.sim.data.time) != before_time or not np.array_equal(before, after):
                        raise RuntimeError("physics changed during the paused generation interval")
                    targets, openings = decode_action_targets(anchor, actions, horizon=config["execution_horizon"])
                    if case["source"] == "oracle":
                        np.testing.assert_allclose(targets, published_tcp[indices[: len(targets)]], atol=1e-12, rtol=0)
                    chunk = dict(
                        sequence=number,
                        step=step,
                        anchor_tcp=anchor,
                        actions=actions,
                        observed_source=measured,
                        targets_tcp=targets,
                        observation_sha256=observation_hash,
                        simulation_time_before=before_time,
                        simulation_time_after=float(env.sim.data.time),
                        state_sha256_before=sha256_bytes(before.tobytes()),
                        state_sha256_after=sha256_bytes(after.tobytes()),
                        generation_wall_s=time.monotonic() - query_started,
                    )
                    chunks.append(chunk)
                    write_json(dest / "chunks.json", chunks)
                    target_index = 0
                desired = targets[target_index] @ inverse(h)
                converted = converter.convert(measured, desired, measured_twist(env))
                command = np.r_[converted["command"], intent.update(float(openings[target_index]))]
                before_time = float(env.sim.data.time)
                cached, _, _, _ = env.step(command)
                obs = current_observation(env)
                after = source_pose(obs)
                row = dict(
                    step=step,
                    chunk_sequence=len(chunks) - 1,
                    target_index=target_index,
                    simulation_time=before_time,
                    simulation_elapsed_s=before_time - initial_time,
                    actual_dt=float(env.sim.data.time) - before_time,
                    measured_source=measured,
                    desired_source=desired,
                    after_source=after,
                    returned_cached_source=source_pose(cached),
                    physical_after_source=transform(
                        env.sim.data.get_site_xpos(env.robots[0].controller.eef_name),
                        env.sim.data.get_body_xmat(env.robots[0].robot_model.eef_name).reshape(3, 3),
                    ),
                    error=pose_error(desired, after),
                    opening_target=float(openings[target_index]),
                    opening_after=aperture(obs),
                    command=command,
                    gripper_internal=gripper.current_action.copy(),
                    success=bool(env.check_success()),
                    contacts=contact_counts(env),
                    bowl_position=env.sim.data.get_body_xpos("akita_black_bowl_1_main").copy(),
                    plate_position=env.sim.data.get_body_xpos("plate_1_main").copy(),
                    saturated=converted["saturated"],
                    rate_limited=converted["rate_limited"],
                )
                rows.append(row)
                stream.write(json.dumps(row, default=lambda v: v.tolist(), allow_nan=False) + "\n")
                stream.flush()
                target_index += 1
                if step % 8 == 7 or step == length - 1:
                    progress = dict(
                        steps=step + 1,
                        chunks=len(chunks),
                        success=any(r["success"] for r in rows),
                        elapsed_s=time.monotonic() - started,
                    )
                    write_json(dest / "progress.json", progress)
                    print(json.dumps(dict(case=case["name"], trial=trial, **progress)), flush=True)
    except Exception as exc:
        error = repr(exc)
    finally:
        env.close()
    result = dict(
        case=case,
        trial=trial,
        episode=record["episode"],
        seed=config["seed"] + trial,
        steps=len(rows),
        expected_steps=length,
        chunks=len(chunks),
        error=error,
        success=any(r["success"] for r in rows),
        terminal_success=bool(rows and rows[-1]["success"]),
        first_success_step=next((r["step"] for r in rows if r["success"]), None),
        wall_elapsed_s=time.monotonic() - started,
        inference_latency_excluded=True,
        every_model_chunk_admitted=True,
        physical_elapsed_s=sum(r["actual_dt"] for r in rows),
    )
    if rows:
        errors = np.asarray([r["error"] for r in rows])
        result.update(
            position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            opening_rmse=float(np.sqrt(np.mean([(r["opening_target"] - r["opening_after"]) ** 2 for r in rows]))),
            final_second_success_fraction=float(np.mean([r["success"] for r in rows[-20:]])),
            finger_contact_steps=sum(r["contacts"]["finger_bowl"] > 0 for r in rows),
            plate_contact_steps=sum(r["contacts"]["bowl_plate"] > 0 for r in rows),
        )
    write_json(dest / "result.json", result)
    if error:
        raise RuntimeError(error)
    return result
