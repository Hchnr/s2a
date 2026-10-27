"""Versioned gripper-position executor with the unchanged recovery framework.

Current experiments keep their old execution source bytes. Shared response,
geometry, authentication and source utilities are reused from that frozen
version; this module adds the explicit position/hold adapter selection. Oracle
qualification and WAM run through the same physical execution function.
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

from s2a.action_reference import decode_action_targets, encode_oracle_targets
from s2a.artifacts import REPO, sha256, write_json
from s2a.cameras import render_current_cameras
from s2a.checkpoint_simulation import object_progress
from s2a.geometry import inverse, pose_error
from s2a.gripper import opening_position_target
from s2a.gripper_intent_hold import GripperIntentHold, GripperIntentHoldConfig
from s2a.gripper_position_hold import GripperPositionHold, GripperPositionHoldConfig
from s2a.gripper_reference_preview import opening_preview
from s2a.gripper_tracking import panda_opening_from_motor_target
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
from s2a.reference_governor import ReferenceGovernorConfig
from s2a.reference_governor_v2 import RecoveringReferenceGovernor
from s2a.stability_recovery_simulation import (
    execution_inputs as recovery_execution_inputs,
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
    """Bind this version and its actual imports, preserving live old sources."""
    signatures = recovery_execution_inputs(config, config_path)
    signatures.pop(str(REPO / "scripts/run_wam_recovery_matrix.py"))
    path = REPO / "scripts/run_wam_position_matrix.py"
    signatures[str(path)] = sha256(path)
    return signatures


def make_gripper(variant, initial_opening, config):
    if variant["gripper"] == "position_hold":
        return GripperPositionHold(initial_opening, config=GripperPositionHoldConfig(**config["gripper_position_hold"]))
    return GripperIntentHold(initial_opening, config=GripperIntentHoldConfig(**config["gripper_intent_hold"]))


def run_gripper_preview_case(
    config,
    root,
    record,
    *,
    source,
    variant_name,
    sampling_seed,
    checkpoint=None,
    config_path,
    frozen_inputs=None,
    oracle_reference,
    endpoint_observer,
):
    """Training-only reference execution; observer records actual endpoints."""
    if source != "oracle" or not callable(endpoint_observer):
        raise ValueError("Training recovery requires explicit oracle references and observer")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("physics requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    if config["gripper_preview_steps"] not in (0, 6):
        raise ValueError("Fixed preview0/6 required")
    variant = position_variant(config, variant_name, source=source)
    if variant["governor"]:
        raise ValueError("Preview pilot requires fixed no-governor interface")
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
    )
    root = Path(root).resolve()
    if not root.is_relative_to(REPO / "outputs"):
        raise ValueError("case outputs must be below repository outputs")
    dest = root / "simulation" / f"{checkpoint or source}_{variant_name}" / f"{record['name']}_s{sampling_seed}"
    dest.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version="s2a.position-execution.v1",
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
        reference, reference_opening, oracle_signatures = oracle_reference
        signatures.update(oracle_signatures)
    h = np.asarray(json.loads(Path(c["tool_contract"]).read_text())["source_from_tcp"])
    h_inv, dt = inverse(h), 1 / c["control_fps"]
    write_json(dest / "manifest.json", manifest)
    env = None
    rows, chunks, failure = [], [], None
    start = time.monotonic()
    try:
        env, _ = make_environment(c, dest, record, 0)
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
        endpoint_observer(env, 0)
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
                preview = opening_preview(full_openings, index - 1, preview_steps=config["gripper_preview_steps"])
                grip_result = gripper.update(
                    preview["reference"],
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
                    command_opening_reference=preview["reference"],
                    gripper_preview=preview,
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
                endpoint_observer(env, step + 1)
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
