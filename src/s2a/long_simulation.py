"""Isolated017 long native and shared oracle/WAM physical executor.

The existing native and position execution loops are copied explicitly, with
the1300 episode horizon and actual1200-step clock evidence added. Their response,
geometry, authentication and source helpers remain frozen. No policy steps are
added during initialization and native references are never padded.
"""

import importlib.metadata
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
from scipy.spatial.transform import Rotation

from s2a.action_reference import decode_action_targets, encode_oracle_targets
from s2a.artifacts import REPO, code_fingerprint, sha256, write_json
from s2a.cameras import render_current_cameras
from s2a.checkpoint_simulation import object_progress
from s2a.geometry import inverse, pose_error
from s2a.gripper import OpeningIntent, opening_position_target
from s2a.gripper_intent_hold import GripperIntentHold, GripperIntentHoldConfig
from s2a.gripper_position_hold import GripperPositionHold, GripperPositionHoldConfig
from s2a.gripper_tracking import panda_opening_from_motor_target
from s2a.libero_simulation import (
    aperture,
    controller_contract,
    current_observation,
    generate,
    measured_twist,
    sha256_bytes,
    source_pose,
)
from s2a.long_environment import episode_contract, episode_snapshot, make_environment, validate_episode_settings
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.reference_governor import ReferenceGovernorConfig
from s2a.reference_governor_v2 import RecoveringReferenceGovernor
from s2a.stability_position_simulation import execution_inputs as position_execution_inputs
from s2a.stability_recovery_simulation import (
    _read_oracle as _read_original_oracle,
)
from s2a.stability_recovery_simulation import (
    sample_model_reference,
    simulator_inputs,
    validate_wam_response,
    verify_inputs,
)
from s2a.task_acceptance import task_outcome
from s2a.wam_regularization import attenuate_wam_rotation


def position_variant(config, name, *, source):
    """Freeze the qualified v2/v3 grippers and explicit model interventions."""
    matches = [v for v in config["variants"] if v["name"] == name]
    if len(matches) != 1 or source not in ("wam", "oracle"):
        raise ValueError("requires one explicit variant and WAM/oracle source")
    variant = dict(matches[0])
    if variant["rotation"] != "response" or variant["gripper"] not in ("intent_hold", "position_hold"):
        raise ValueError("only response inverse and the explicit v2/v3 grippers are admitted")
    if variant["governor"] is not False and variant["governor"] != "recovering":
        raise ValueError("governor must be false or recovering")
    alpha = variant["model_rotation_alpha"]
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float)) or alpha not in (0, 0.5, 1):
        raise ValueError("model_rotation_alpha must explicitly be 1, .5 or 0")
    if source == "oracle" and alpha != 1:
        raise ValueError("oracle qualification must preserve its supplied reference")
    if asdict(GripperIntentHoldConfig(**config["gripper_intent_hold"])) != asdict(GripperIntentHoldConfig()):
        raise ValueError("this executor requires the physically qualified intent_hold defaults")
    if asdict(GripperPositionHoldConfig(**config["gripper_position_hold"])) != asdict(GripperPositionHoldConfig()):
        raise ValueError("position adapter parameters differ from the frozen qualification defaults")
    ReferenceGovernorConfig(**config["reference_governor"])
    return variant


def execution_inputs(config, config_path):
    """Retain old helper inputs and bind the explicit017 entry/factory."""
    signatures = position_execution_inputs(config, config_path)
    for relative in ("src/s2a/long_environment.py", "src/s2a/long_simulation.py", "scripts/run_wam_long_matrix.py"):
        path = REPO / relative
        signatures[str(path)] = sha256(path)
    return signatures


def make_gripper(variant, initial_opening, config):
    if variant["gripper"] == "position_hold":
        return GripperPositionHold(initial_opening, config=GripperPositionHoldConfig(**config["gripper_position_hold"]))
    return GripperIntentHold(initial_opening, config=GripperIntentHoldConfig(**config["gripper_intent_hold"]))


def run_long_position_case(
    config, root, record, *, source, variant_name, sampling_seed, checkpoint=None, config_path, frozen_inputs=None
):
    """Run one fresh CPU environment; WAM inference pauses physics, not its budget."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("physics requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    validate_long_config(config, source=source)
    variant = position_variant(config, variant_name, source=source)
    signatures = execution_inputs(config, config_path)
    if frozen_inputs is not None:
        verify_inputs(frozen_inputs)
        if any(frozen_inputs.get(path) != digest for path, digest in signatures.items()):
            raise ValueError("worker execution inputs differ from coordinator snapshot")
        signatures = dict(frozen_inputs)
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
        episode_horizon=config["episode_horizon"],
        ignore_done=config["ignore_done"],
    )
    root = Path(root).resolve()
    if not root.is_relative_to(REPO / "outputs"):
        raise ValueError("case outputs must be below repository outputs")
    dest = root / "simulation" / f"{checkpoint or source}_{variant_name}" / f"{record['name']}_s{sampling_seed}"
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version="s2a.position-execution.v1",
        long_episode_contract=dict(
            episode_horizon=1300, ignore_done=False, physical_steps=1200, initialization_policy_steps_added=0
        ),
        implementation_origin="stability_recovery_simulation.py at d8421f9; explicit position/hold adapter version",
        execution_sha256=signatures,
        experiment=config,
        runtime_config=c,
        record=record,
        source=source,
        variant=variant,
        inference_latency_excluded=True,
        oracle_is_wam_success_evidence=False,
        raw_checkpoint_reference=bool(source == "wam" and variant["model_rotation_alpha"] == 1),
        reference_intervention="raw" if variant["model_rotation_alpha"] == 1 else "model_rotation_attenuation",
        online_control_inputs="current measured TCP/twist, motor target, aperture/history and generated references",
        object_contact_and_task_predicate="metrics only; never adapter/governor inputs",
        oracle_reference_mode="fixed_world_chunks_selected_by_current_physical_step" if source == "oracle" else None,
        oracle_qualification_scope=(
            "privileged physical-step indexed chunks; differs from prior one-shot full600 reference qualification"
            if source == "oracle"
            else None
        ),
        environment={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "MUJOCO_GL")},
        git_commit_metadata=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    )
    expected_checkpoint = None
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
        expected_checkpoint = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
        if (
            Path(info["checkpoint_path"]).resolve() != expected_checkpoint.resolve()
            or info["ema_weights"]
            or not info["acceptance_intervention"]["active_slot_mask"]
        ):
            raise ValueError("wrong model or missing inactive-slot mask")
        manifest["server_info"] = info
    else:
        if checkpoint is not None:
            raise ValueError("oracle cannot be labeled a WAM checkpoint")
        reference, reference_opening, oracle_signatures = _read_oracle(config, record, sampling_seed)
        signatures.update(oracle_signatures)
    h = np.asarray(json.loads(Path(c["tool_contract"]).read_text())["source_from_tcp"])
    h_inv, dt = inverse(h), 1 / c["control_fps"]
    write_json(dest / "manifest.json", manifest)
    env = None
    initial_episode = None
    rows, chunks, failure = [], [], None
    start = time.monotonic()
    try:
        env, _ = make_environment(c, dest, record, 0)
        initial_episode = episode_snapshot(env)
        obs = current_observation(env)
        controller, contract = controller_contract(env)
        q = source_pose(obs)[:3, :3].T @ np.asarray(env.sim.data.get_site_xmat(controller.eef_name))
        if not np.isclose(env.env.control_timestep, dt):
            raise ValueError("control interval differs from declared model time")
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
        native_initial_motor = np.asarray(env.robots[0].gripper.current_action).copy()
        env.robots[0].gripper.current_action = opening_position_target(initial_opening)
        gripper = make_gripper(variant, initial_opening, config)
        governor = RecoveringReferenceGovernor(config=ReferenceGovernorConfig(**config["reference_governor"]))
        initial_objects = object_progress(env)["objects"]
        simulator_signatures = simulator_inputs(env)
        verify_inputs(signatures)
        signatures.update(simulator_signatures)
        manifest["execution_sha256"] = signatures
        manifest["simulator_source_sha256"] = simulator_signatures
        model_xml = env.sim.model.get_xml()
        (dest / "model.xml").write_text(model_xml)
        manifest["initial_model_xml_sha256"] = sha256_bytes(model_xml.encode())
        manifest["package_versions"] = {}
        for package in ("numpy", "scipy", "pyarrow", "requests", "mujoco", "robosuite", "torch", "pillow"):
            try:
                manifest["package_versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                manifest["package_versions"][package] = None
        write_json(dest / "manifest.json", manifest)
        write_json(
            dest / "controller.json",
            dict(
                contract=contract,
                source_from_control_rotation=q,
                native_initial_motor_target=native_initial_motor,
                aligned_initial_motor_target=env.robots[0].gripper.current_action.copy(),
                motor_initialization="observed aperture alignment, not hidden expert motor state",
                gripper_config=asdict(gripper.config),
                actual_initial_physics_sha256=sha256_bytes(np.asarray(env.sim.get_state().flatten()).tobytes()),
            ),
        )
        previous_command = np.zeros(7)
        targets, openings, index, chunk_step, governed = None, None, 0, 0, None
        with (dest / "steps.jsonl").open("x") as stream:
            for step in range(config["online_max_steps"]):
                measured, opening = source_pose(obs), aperture(obs)
                measured_tcp = measured @ h
                need_query = targets is None
                if not need_query:
                    need_query = (
                        bool(governed["complete"]) if variant["governor"] else index >= len(targets)
                    ) or step - chunk_step >= config["max_observation_interval_steps"]
                if need_query:
                    before = np.asarray(env.sim.get_state().flatten()).copy()
                    before_time, query_start = float(env.sim.data.time), time.monotonic()
                    query_anchor, query_opening = measured_tcp.copy(), opening
                    identity = None
                    if source == "wam":
                        raw, image_sha = generate(
                            c, render_current_cameras(env, c["render_resolution"]), dest, len(chunks)
                        )
                        response_path = dest / f"response_{len(chunks):04d}.json"
                        response = json.loads(response_path.read_text())
                        authenticated, identity = validate_wam_response(
                            response,
                            checkpoint_path=expected_checkpoint,
                            seed=sampling_seed,
                            domain=c["action_domain"],
                            fps=c["control_fps"],
                            horizon=config["online_chunk_size"],
                        )
                        if not np.array_equal(raw, authenticated):
                            raise ValueError("decoded generator action differs from saved response")
                        effective = attenuate_wam_rotation(raw, alpha=variant["model_rotation_alpha"])
                    else:
                        end = min(step + config["online_chunk_size"], len(reference))
                        fixed_targets = reference[step:end] @ h
                        fixed_openings = reference_opening[step:end]
                        raw = encode_oracle_targets(query_anchor, fixed_targets, fixed_openings)
                        effective, image_sha = raw.copy(), None
                    full_raw_targets, full_openings = decode_action_targets(query_anchor, raw, horizon=len(raw))
                    full_effective_targets, _ = decode_action_targets(query_anchor, effective, horizon=len(effective))
                    count = min(config["online_execution_horizon"], len(raw))
                    raw_targets, targets, openings = (
                        full_raw_targets[:count],
                        full_effective_targets[:count],
                        full_openings[:count],
                    )
                    after = np.asarray(env.sim.get_state().flatten()).copy()
                    after_time = float(env.sim.data.time)
                    if after_time != before_time or not np.array_equal(before, after):
                        raise RuntimeError("physics advanced while generating/transforming a reference")
                    summary = None
                    if variant["governor"]:
                        summary = governor.set_reference(
                            query_anchor, targets, openings, anchor_opening=opening, model_dt=dt
                        )
                    index, chunk_step = 0, step
                    chunks.append(
                        dict(
                            step=step,
                            anchor_tcp=query_anchor,
                            anchor_opening=query_opening,
                            raw128=raw,
                            effective128=effective,
                            raw_targets_tcp=full_raw_targets,
                            effective_targets_tcp=full_effective_targets,
                            openings=full_openings,
                            execution_endpoint_count=count,
                            model_rotation_alpha=variant["model_rotation_alpha"],
                            response_identity=identity,
                            request_sha256=sha256(dest / f"request_{len(chunks):04d}.json")
                            if source == "wam"
                            else None,
                            response_sha256=sha256(response_path) if source == "wam" else None,
                            image_sha256=image_sha,
                            physics_before_sha256=sha256_bytes(before.tobytes()),
                            physics_after_sha256=sha256_bytes(after.tobytes()),
                            physics_unchanged=True,
                            physics_before_time_s=before_time,
                            physics_after_time_s=after_time,
                            generation_s=time.monotonic() - query_start,
                            retiming=summary,
                        )
                    )
                    write_json(dest / "chunks.json", chunks)
                if variant["governor"]:
                    governed = governor.advance(measured_tcp, dt)
                    desired_tcp, desired_opening = governed["target_tcp"], governed["opening"]
                    model_time = governed["model_time_s"]
                else:
                    desired_tcp, desired_opening = targets[index], float(openings[index])
                    index += 1
                    model_time = index * dt
                raw_desired, raw_opening = sample_model_reference(
                    query_anchor,
                    raw_targets,
                    openings,
                    model_time_s=model_time,
                    dt=dt,
                    anchor_opening=query_opening,
                )
                effective_desired, effective_opening = sample_model_reference(
                    query_anchor,
                    targets,
                    openings,
                    model_time_s=model_time,
                    dt=dt,
                    anchor_opening=query_opening,
                )
                desired = desired_tcp @ h_inv
                twist = measured_twist(env)
                goal_before, goal_pos_before = (
                    np.asarray(controller.goal_ori).copy(),
                    np.asarray(controller.goal_pos).copy(),
                )
                converted = converter.convert(measured, desired, twist)
                motor_before = np.asarray(env.robots[0].gripper.current_action).copy()
                motor_opening = panda_opening_from_motor_target(motor_before)
                grip_result = gripper.update(
                    desired_opening,
                    opening,
                    motor_opening=motor_opening,
                    previous_command=previous_command[-1],
                    dt=dt,
                )
                grip_command = grip_result.command
                command = np.r_[converted["command"], grip_command]
                gripper_qpos_before = np.asarray(obs["robot0_gripper_qpos"]).copy()
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
                    chunk_physical_step=step - chunk_step,
                    physical_time_before_s=before_time,
                    physical_time_after_s=float(env.sim.data.time),
                    measured_source=measured,
                    measured_twist=twist,
                    desired_source=desired,
                    desired_tcp=desired_tcp,
                    raw_desired_tcp=raw_desired,
                    effective_desired_tcp=effective_desired,
                    raw_opening_at_model_time=raw_opening,
                    effective_opening_at_model_time=effective_opening,
                    model_time_s=model_time,
                    retimed_clock_s=governed["retimed_clock_s"] if governed else model_time,
                    bridging=bool(governed["bridging"]) if governed else False,
                    bridge_recovery_advanced=bool(governed["bridge_recovery_advanced"]) if governed else False,
                    after_source=after_pose,
                    opening_target=desired_opening,
                    opening_before=opening,
                    opening_after=aperture(obs),
                    gripper_qpos_before=gripper_qpos_before,
                    gripper_qpos_after=np.asarray(obs["robot0_gripper_qpos"]).copy(),
                    motor_opening_before=motor_opening,
                    motor_opening_after=panda_opening_from_motor_target(env.robots[0].gripper.current_action),
                    motor_target_before=motor_before,
                    motor_target_after=env.robots[0].gripper.current_action.copy(),
                    actual_gripper_actuator_control=env.sim.data.ctrl[
                        env.robots[0]._ref_joint_gripper_actuator_indexes
                    ].copy(),
                    controller_goal_before=goal_before,
                    controller_goal_after=np.asarray(controller.goal_ori).copy(),
                    controller_goal_position_before=goal_pos_before,
                    controller_goal_position_after=np.asarray(controller.goal_pos).copy(),
                    controller_goal_source_rotation_before=goal_before @ q.T,
                    command=command,
                    actual_dt=float(env.sim.data.time) - before_time,
                    source_error=pose_error(desired, after_pose),
                    tcp_error=pose_error(desired_tcp, after_pose @ h),
                    raw_model_tcp_error=pose_error(raw_desired, after_pose @ h),
                    effective_model_tcp_error=pose_error(effective_desired, after_pose @ h),
                    reference_progress=governed,
                    gripper=asdict(grip_result) if grip_result else None,
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
        if env is not None:
            if initial_episode is not None:
                contract_clock = episode_contract(initial_episode, episode_snapshot(env), len(rows))
                write_json(dest / "episode_contract.json", contract_clock)
                if not contract_clock["clock_consistent"]:
                    failure = f"{failure + '; ' if failure else ''}long episode clock inconsistent"
            env.close()
    input_failure = None
    try:
        verify_inputs(signatures)
    except BaseException as exc:
        input_failure = repr(exc)
        failure = f"{failure + '; ' if failure else ''}{exc!r}"
    complete = len(rows) == config["online_max_steps"] and failure is None
    outcome = (
        task_outcome(
            [r["success"] for r in rows],
            expected_steps=config["online_max_steps"],
            stable_steps=config["stable_success_last_steps"],
        )
        if complete
        else None
    )
    result = dict(
        record=record["name"],
        task=record["task"],
        source=source,
        checkpoint=checkpoint,
        variant=variant_name,
        sampling_seed=sampling_seed,
        model_rotation_alpha=variant["model_rotation_alpha"],
        raw_checkpoint_reference=manifest["raw_checkpoint_reference"],
        steps=len(rows),
        expected_steps=config["online_max_steps"],
        error=failure,
        complete=complete,
        chunks=len(chunks),
        elapsed_s=time.monotonic() - start,
        inference_latency_excluded=True,
        oracle_is_wam_success_evidence=False,
        stable_success=bool(outcome and outcome["stable"]),
        ever_success=any(r["success"] for r in rows),
        task_outcome=outcome,
        execution_inputs_unchanged=input_failure is None,
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


def _run_native_case(
    config,
    root,
    record,
    *,
    source,
    sampling_seed,
    checkpoint=None,
    probe=None,
    server_port=None,
    config_path,
    frozen_inputs=None,
):
    validate_long_config(config, source=source)
    if source != "native" or checkpoint is not None or probe is not None or server_port is not None:
        raise ValueError("this literal legacy branch is restricted to017 native")
    signatures = execution_inputs(config, config_path)
    if frozen_inputs is not None:
        verify_inputs(frozen_inputs)
        if any(frozen_inputs.get(path) != digest for path, digest in signatures.items()):
            raise ValueError("native worker inputs differ from coordinator")
        signatures = dict(frozen_inputs)
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
        episode_horizon=config["episode_horizon"],
        ignore_done=config["ignore_done"],
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
        if native.ndim != 2 or native.shape[1] != 7 or not len(native) or not np.isfinite(native).all():
            raise ValueError("invalid native expert commands")
        signatures[str(Path(record["expert_path"]).resolve())] = record["expert_sha256"]
    if source == "oracle":
        reference_path = root / "simulation/native" / f"{record['name']}_s{sampling_seed}/reference.npz"
        manifest["reference_sha256"] = sha256(reference_path)
        with np.load(reference_path) as data:
            reference, opening_reference = data["poses"], data["openings"]
    manifest.update(
        schema_version="s2a.long-native-execution.v1",
        experiment=config,
        execution_sha256=signatures,
        long_episode_contract=dict(
            episode_horizon=1300, ignore_done=False, physical_steps=1200, initialization_policy_steps_added=0
        ),
    )
    write_json(dest / "manifest.json", manifest)
    env, _ = make_environment(c, dest, record, 0)
    initial_episode = episode_snapshot(env)
    simulator_signatures = simulator_inputs(env)
    signatures.update(simulator_signatures)
    manifest["execution_sha256"] = signatures
    manifest["simulator_source_sha256"] = simulator_signatures
    model_xml = env.sim.model.get_xml()
    (dest / "model.xml").write_text(model_xml)
    manifest["initial_model_xml_sha256"] = sha256_bytes(model_xml.encode())
    manifest["actual_initial_physics_sha256"] = sha256_bytes(np.asarray(env.sim.get_state().flatten()).tobytes())
    write_json(dest / "manifest.json", manifest)
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
        contract_clock = episode_contract(initial_episode, episode_snapshot(env), len(rows))
        write_json(dest / "episode_contract.json", contract_clock)
        if not contract_clock["clock_consistent"]:
            error = f"{error + '; ' if error else ''}long episode clock inconsistent"
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
    input_failure = None
    try:
        verify_inputs(signatures)
    except BaseException as exc:
        input_failure = repr(exc)
        error = f"{error + '; ' if error else ''}{exc!r}"
    complete = len(rows) == count and error is None
    outcome = (
        task_outcome(
            [row["success"] for row in rows], expected_steps=count, stable_steps=config["stable_success_last_steps"]
        )
        if complete
        else None
    )
    result.update(
        error=error,
        complete=complete,
        task_outcome=outcome,
        stable_success=bool(outcome and outcome["stable"]),
        execution_inputs_unchanged=input_failure is None,
    )
    write_json(dest / "result.json", result)
    if error:
        raise RuntimeError(error)
    return result


def _read_oracle(config, record, sampling_seed):
    """Admit only this registered native1200, never a padded600 reference."""
    base = Path(config["oracle_reference_root"]).resolve()
    if not base.is_relative_to(REPO / "outputs") or (base / record["name"] / "reference.npz").exists():
        raise ValueError("017 requires an unambiguous repository native/record_seed reference")
    folder = base / f"{record['name']}_s{sampling_seed}"
    manifest = json.loads((folder / "manifest.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    clock = json.loads((folder / "episode_contract.json").read_text())
    if (
        manifest["record"] != record
        or manifest["source"] != "native"
        or manifest["schema_version"] != "s2a.long-native-execution.v1"
        or result["record"] != record["name"]
        or result["source"] != "native"
        or result["sampling_seed"] != sampling_seed
        or not result["complete"]
        or not result["execution_inputs_unchanged"]
        or result["steps"] != 1200
        or not clock["clock_consistent"]
        or clock["effective_horizon"] != 1300
        or clock["ignore_done"] is not False
        or clock["physical_steps"] != 1200
    ):
        raise ValueError("017 oracle source is not the actual matching native1200")
    poses, openings, signatures = _read_original_oracle(config, record, sampling_seed)
    if len(poses) != 1200 or len(openings) != 1200:
        raise ValueError("017 must not extend a short native reference")
    for name in ("manifest.json", "episode_contract.json"):
        path = folder / name
        signatures[str(path)] = sha256(path)
    return poses, openings, signatures


def validate_long_config(config, *, source):
    """Only the registered017 duration/raw8 variant is admitted."""
    validate_episode_settings(config)
    expected = dict(
        online_max_steps=1200,
        online_chunk_size=40,
        online_execution_horizon=40,
        max_observation_interval_steps=8,
        stable_success_last_steps=40,
        control_fps=20,
        sampling_steps=35,
        guidance=1,
        shift=5,
        inverse_weight=0.5,
        max_command_rate=5,
    )
    if any(
        isinstance(config.get(k), (bool, np.bool_)) or config.get(k) != value for k, value in expected.items()
    ) or config.get("sampling_seeds") != [42]:
        raise ValueError("017 duration/sampling/controller contract differs")
    if source not in ("native", "oracle", "wam") or config.get("variants") != [
        dict(
            name="position_hold", rotation="response", gripper="position_hold", governor=False, model_rotation_alpha=1.0
        )
    ]:
        raise ValueError("017 preserves raw no-governor position/hold")


def run_long_case(
    config, root, record, *, source, variant_name, sampling_seed, checkpoint=None, config_path, frozen_inputs=None
):
    """Dispatch native positive control or the shared017 oracle/WAM executor."""
    validate_long_config(config, source=source)
    if type(sampling_seed) is not int or sampling_seed != 42 or variant_name != "position_hold":
        raise ValueError("017 only admits registered position_hold/seed42")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("017 native and physics require empty CUDA/LD")
    root = Path(root).resolve()
    if not root.is_relative_to(REPO / "outputs") or root == REPO / "outputs":
        raise ValueError("017 case outputs must belong to a named repository outputs directory")
    if source == "native":
        return _run_native_case(
            config,
            root,
            record,
            source=source,
            sampling_seed=sampling_seed,
            checkpoint=checkpoint,
            config_path=config_path,
            frozen_inputs=frozen_inputs,
        )
    return run_long_position_case(
        config,
        root,
        record,
        source=source,
        variant_name=variant_name,
        sampling_seed=sampling_seed,
        checkpoint=checkpoint,
        config_path=config_path,
        frozen_inputs=frozen_inputs,
    )
