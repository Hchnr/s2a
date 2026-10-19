"""Continuous MuJoCo feedback with delayed oracle forecasts or live WAM HTTP.

Synthetic forecasts use privileged, already-generated reference trajectories.
Live mode reads only the fixture initial state, current rendered observations,
and completed model responses. It never loads expert trajectories.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.async_reference import AsyncReferenceBuffer, TimedReference
from s2a.cameras import render_current_cameras
from s2a.geometry import inverse, pose_error, transform
from s2a.gripper import OpeningIntent, opening_position_target
from s2a.libero_simulation import (
    aperture,
    controller_contract,
    current_observation,
    generate,
    make_environment,
    measured_twist,
    source_pose,
)
from s2a.osc import OSCInterface, OSCStateToAction


def case_name(stage, arm, delay):
    return f"{stage}_{arm}" + (f"_delay{round(delay * 1000)}ms" if stage == "synthetic" else "")


def run_async_case(config, output, *, stage, arm, trial, delay=0.0):
    if stage not in ("synthetic", "live") or arm not in ("affine", "response"):
        raise ValueError("unknown asynchronous experiment")
    if trial not in config["trials"] or not np.isfinite(delay) or delay < 0 or (stage == "live" and delay != 0):
        raise ValueError("invalid trial or inference delay")
    name = case_name(stage, arm, delay)
    dest = output / "async" / name / f"trial{trial}"
    dest.mkdir(parents=True, exist_ok=False)
    policy = json.loads((output / "serving_config.json").read_text())
    fixture = json.loads(Path(policy["fixture"]).read_text())
    record = fixture["episodes"][trial]
    h = np.asarray(json.loads(Path(policy["tool_contract"]).read_text())["source_from_tcp"])
    dt = 1 / policy["control_fps"]
    horizon = policy["execution_horizon"]
    inputs = {str(p): sha256(p) for p in (policy["fixture"], policy["tool_contract"], output / "serving_config.json")}
    manifest = dict(config=config, stage=stage, arm=arm, trial=trial, delay_s=delay, code=code_fingerprint())
    if stage == "synthetic":
        bridge_path = config["published_bridge"]
        bridge = json.loads(Path(bridge_path).read_text())["trajectories"][record["demo"]]
        poses = np.asarray(bridge["source_poses"]) @ h
        openings = np.asarray(bridge["openings"])
        inputs[bridge_path] = sha256(bridge_path)
        length = max(config["synthetic_min_steps"], len(poses) + policy["oracle_extra_hold_steps"])
        manifest["evidence"] = "Independent MuJoCo with delayed privileged published-reference forecasts; not WAM"
    else:
        import requests

        with requests.Session() as session:
            session.trust_env = False
            response = session.get(policy["server_url"] + "/info", timeout=15)
            response.raise_for_status()
            info = response.json()
        expected = Path(policy["checkpoint_run"]) / "checkpoints" / policy["checkpoint_iteration"] / "model"
        if Path(info["checkpoint_path"]).resolve() != expected.resolve():
            raise ValueError("server loaded a different checkpoint")
        manifest["server_info"] = info
        manifest["evidence"] = "Live-image WAM and continuous MuJoCo feedback; wall-clock reference expiry"
        length = config["async_steps"]
    manifest["inputs_sha256"] = inputs
    write_json(dest / "manifest.json", manifest)
    env, obs = make_environment({**policy, "camera_observations": False}, dest, record, trial)
    rows, events = [], []
    error, pending, executor = None, None, None
    started = time.monotonic()
    sequence, initial_sim_time = 0, float(env.sim.data.time)
    try:
        obs = current_observation(env)
        _, contract = controller_contract(env)
        if not np.isclose(env.env.control_timestep, dt):
            raise ValueError("control interval differs from reference period")
        converter = OSCStateToAction(
            OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max"))),
            contract["kp"],
            contract["kd"],
            dt,
            compensate=arm == "response",
            inverse_weight=policy["inverse_weight"],
            max_command_rate=policy["max_command_rate"],
        )
        gripper = env.robots[0].gripper
        gripper.current_action = opening_position_target(aperture(obs))
        intent = OpeningIntent(aperture(obs), band=policy["opening_hysteresis"])
        buffer = AsyncReferenceBuffer(source_pose(obs) @ h, aperture(obs), max_age=config["async_max_age_s"])
        camera_warmup_s = 0.0
        if stage == "live":
            warmup_started = time.monotonic()
            render_current_cameras(env, policy["render_resolution"])
            camera_warmup_s = time.monotonic() - warmup_started
            executor = ThreadPoolExecutor(max_workers=1)
        write_json(
            dest / "controller.json",
            dict(**contract, initial_gripper=gripper.current_action.copy(), camera_warmup_s=camera_warmup_s),
        )
        started = time.monotonic()
        next_tick_wall = started
        initial_sim_time = float(env.sim.data.time)
        sequence, last_request = 0, -np.inf

        def query(snapshot, number):
            actions, image_hash = generate(policy, snapshot, dest, number)
            return actions, image_hash, time.monotonic() - started

        def admit(meta, reference, now, **extra):
            verdict = buffer.admit(reference, now=now)
            event = dict(**meta, admitted_at=now, age_s=now - meta["observed_at"], verdict=verdict, **extra)
            events.append(event)
            write_json(dest / "events.json", events)

        with (dest / "steps.jsonl").open("w") as stream:
            for step in range(length):
                now = step * dt if stage == "synthetic" else time.monotonic() - started
                current = source_pose(obs)
                if pending is not None:
                    if stage == "synthetic" and now + 1e-9 >= pending[0]:
                        _, meta, reference = pending
                        admit(meta, reference, now)
                        pending = None
                    elif stage == "live" and pending[0].done():
                        future, meta, anchor = pending
                        actions, image_hash, completed = future.result()
                        reference = TimedReference.from_actions(
                            meta["sequence"], meta["observed_at"], dt, anchor, actions[:, :10], horizon=horizon
                        )
                        now = time.monotonic() - started
                        admit(meta, reference, now, completed_at=completed, observation_sha256=image_hash)
                        pending = None
                interval = config["synthetic_request_interval_steps"] * dt
                if pending is None and now - last_request >= interval - 1e-9:
                    meta = dict(sequence=sequence, capture_step=step, observed_at=now)
                    if stage == "synthetic":
                        indices = np.minimum(step + np.arange(horizon), len(poses) - 1)
                        reference = TimedReference(sequence, now, dt, poses[indices], openings[indices])
                        if delay == 0:
                            admit(meta, reference, now)
                        else:
                            pending = now + delay, meta, reference
                    else:
                        # Include current-image acquisition in observation age.
                        snapshot = render_current_cameras(env, policy["render_resolution"])
                        meta["capture_duration_s"] = time.monotonic() - started - now
                        pending = executor.submit(query, snapshot, sequence), meta, current @ h
                    sequence += 1
                    last_request = now
                now = step * dt if stage == "synthetic" else time.monotonic() - started
                target = buffer.sample(now=now)
                desired = target["pose"] @ inverse(h)
                converted = converter.convert(current, desired, measured_twist(env))
                grip_command = intent.update(target["opening"]) if target["tracking"] else 0.0
                command = np.r_[converted["command"], grip_command]
                before_time = float(env.sim.data.time)
                grip_before = gripper.current_action.copy()
                obs, _, _, _ = env.step(command)
                cached = source_pose(obs)
                obs = current_observation(env)
                after = source_pose(obs)
                row = dict(
                    step=step,
                    clock_s=now,
                    wall_elapsed_s=time.monotonic() - started,
                    simulation_elapsed_s=before_time - initial_sim_time,
                    actual_dt=float(env.sim.data.time) - before_time,
                    measured_source=current,
                    desired_source=desired,
                    after_source=after,
                    returned_cached_source=cached,
                    physical_after_source=transform(
                        env.sim.data.get_site_xpos(env.robots[0].controller.eef_name),
                        env.sim.data.get_body_xmat(env.robots[0].robot_model.eef_name).reshape(3, 3),
                    ),
                    error=pose_error(desired, after),
                    opening_target=target["opening"],
                    opening_after=aperture(obs),
                    command=command,
                    gripper_before=grip_before,
                    gripper_after=gripper.current_action.copy(),
                    target_status=target["reason"],
                    reference_sequence=target["sequence"],
                    reference_index=target["reference_index"],
                    reference_observed_at=buffer.reference.observed_at if buffer.reference is not None else None,
                    success=bool(env.check_success()),
                    saturated=converted["saturated"],
                    rate_limited=converted["rate_limited"],
                )
                if stage == "live":
                    row["planned_start_clock_s"] = next_tick_wall - started
                    row["deadline_lateness_s"] = max(0.0, time.monotonic() - next_tick_wall - dt)
                rows.append(row)
                stream.write(json.dumps(row, default=lambda value: value.tolist(), allow_nan=False) + "\n")
                stream.flush()
                if stage == "live":
                    # Physics and feedback continue while the HTTP worker runs.
                    # Deadline misses remain measured; no catch-up steps are hidden.
                    next_tick_wall = max(next_tick_wall + dt, time.monotonic())
                    remaining = next_tick_wall - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                if step % 100 == 0:
                    write_json(
                        dest / "progress.json",
                        dict(
                            step=step + 1,
                            events=len(events),
                            **{k: row[k] for k in ("clock_s", "target_status", "success")},
                        ),
                    )
    except Exception as exc:
        error = repr(exc)
    finally:
        physical_stop_wall_s = time.monotonic() - started
        if executor is not None:
            executor.shutdown(wait=True)
            if pending is not None:
                future, meta, _ = pending
                try:
                    _, image_hash, completed = future.result()
                    events.append(
                        dict(
                            **meta,
                            verdict="not_admitted_shutdown",
                            completed_at=completed,
                            observation_sha256=image_hash,
                        )
                    )
                except Exception as exc:
                    error = error or repr(exc)
        elif pending is not None:
            events.append(dict(**pending[1], verdict="not_arrived_before_shutdown"))
        env.close()
    write_json(dest / "events.json", events)
    result = dict(
        stage=stage,
        arm=arm,
        trial=trial,
        delay_s=delay,
        seed=config["seed"] + trial,
        error=error,
        steps=len(rows),
        expected_steps=length,
        requests=sequence,
        accepted=sum(e["verdict"] == "accepted" for e in events),
        expired=sum(e["verdict"] == "expired" for e in events),
        tracking_steps=sum(r["target_status"] == "tracking" for r in rows),
        success=any(r["success"] for r in rows),
        terminal_success=bool(rows and rows[-1]["success"]),
        physical_stop_wall_s=physical_stop_wall_s,
        total_elapsed_s=time.monotonic() - started,
        initial_sim_time=initial_sim_time,
    )
    if rows:
        errors = np.asarray([r["error"] for r in rows])
        result.update(
            target_position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
            target_rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
            physical_elapsed_s=sum(r["actual_dt"] for r in rows),
            max_deadline_lateness_s=max(r.get("deadline_lateness_s", 0) for r in rows),
        )
    write_json(dest / "result.json", result)
    if error:
        raise RuntimeError(error)
    return result
