"""Read-only006 audit: exactly12 WAM trials, with prior004 qualification referenced.

The literal verify_case fork retains every140f7e8 step check. Its only changed
call binds the first query to the same8400 old recovery batch; no dynamic patch
or modification of the original33-case checker is used.
"""

import argparse
import ast
import difflib
import hashlib
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_earlier_recovery import process_environment
from audit_wam_position import (
    REPO,
    TARGETS,
    Evidence,
    GripperIntentHold,
    GripperIntentHoldConfig,
    GripperPositionHold,
    GripperPositionHoldConfig,
    GripperReplay,
    OSCInterface,
    OSCStateToAction,
    RecoveryReplay,
    Rotation,
    asdict,
    close,
    decode,
    error,
    mapping,
    native_delta_rotation,
    rmse,
    sample,
    task_outcome,
    verify_004_gate,
    verify_query,
)

CONFIG = "configs/wam_position_pilot_8400_g10_20261021.json"
BASE = "scripts/audit_wam_position.py"
BASE_SHA = "8ab8e29b90f6e0419171e563192c2009ce6220d8b5a9c642699798f6e2f19915"
OLD_AUDIT = "outputs/wam_stability_20261021/earlier_recovery_final_audit.json"
OLD_AUDIT_SHA = "aafb3a0aed076ecaeb669445c4c535f96088445d2e39e110c811d11c4d0d567d"
VARIANTS = (
    "position_hold",
    "recovering_position_hold",
    "recovering_position_hold_alpha05",
    "recovering_position_hold_alpha0",
)
SCHEMA = "s2a.earlier-position-readonly-audit.v1"


def verify_first_query006(config, record, variant, case, chunk, evidence, checkpoint):
    names = dict(
        zip(
            VARIANTS,
            (
                "intent_hold",
                "recovering_intent_hold",
                "recovering_intent_hold_alpha05",
                "recovering_intent_hold_alpha0",
            ),
            strict=True,
        )
    )
    assert checkpoint == "8400_masked" and chunk["step"] == 0
    previous = (
        REPO
        / "outputs/wam_stability_20261021/recovery_earlier_development/8400_g10/simulation"
        / f"8400_masked_{names[variant['name']]}"
        / f"{record['name']}_s42"
    )
    evidence.check(OLD_AUDIT, OLD_AUDIT_SHA)
    old_audit = evidence.json(OLD_AUDIT)
    assert old_audit["complete"] and old_audit["all12_cases_independently_verified"]
    for filename in ("input_0000.png", "request_0000.json", "response_0000.json", "chunks.json"):
        path = previous / filename
        evidence.check(path, old_audit["source_before_after_hashes"][str(path)])
    old_request, request = evidence.json(previous / "request_0000.json"), evidence.json(case / "request_0000.json")
    for key in (
        "model_mode",
        "seed",
        "fps",
        "num_frames",
        "action_chunk_size",
        "guidance",
        "num_steps",
        "shift",
        "prompt",
        "extra",
    ):
        assert old_request[key] == request[key], key
    old_content = evidence.json(previous / "response_0000.json")["outputs"][0]["content"]
    content = evidence.json(case / "response_0000.json")["outputs"][0]["content"]
    old_chunk = evidence.json(previous / "chunks.json")[0]
    np.testing.assert_array_equal(old_content["action"], content["action"])
    np.testing.assert_array_equal(old_content["action"], old_chunk["raw128"])
    np.testing.assert_array_equal(content["action"], chunk["raw128"])
    np.testing.assert_array_equal(
        old_content["action128"]["normalized_action"], content["action128"]["normalized_action"]
    )
    close(old_chunk["anchor_tcp"], chunk["anchor_tcp"], tolerance=0)
    assert old_chunk["physics_before_sha256"] == chunk["physics_before_sha256"]
    assert evidence.read(previous / "input_0000.png") == evidence.read(case / "input_0000.png")
    return dict(
        source_case=str(previous),
        image_png_bytes_exact=True,
        full40x128_raw_exact=True,
        full40x128_normalized_exact=True,
        fixed_initial_anchor_exact=True,
        initial_physics_sha_exact=True,
        model_conditions_exact=True,
        scope="006 first query versus corresponding8400 old12 recovery case; later causal observations may differ",
    )


def verify_case(config, record, variant, case, evidence, *, source, checkpoint, full):
    result = evidence.json(case / "result.json")
    manifest = evidence.json(case / "manifest.json")
    controller = evidence.json(case / "controller.json")
    chunks = evidence.json(case / "chunks.json")
    rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
    assert len(rows) == result["steps"] == result["expected_steps"] == config["online_max_steps"] == 600
    assert result["error"] is None and result["complete"] and result["execution_inputs_unchanged"]
    assert manifest["source"] == result["source"] == source
    assert manifest["experiment"] == config and manifest["variant"] == variant and manifest["record"] == record
    assert result["sampling_seed"] == 42 and result["variant"] == variant["name"]
    assert result["checkpoint"] == checkpoint and result["model_rotation_alpha"] == variant["model_rotation_alpha"]
    raw_flag = source == "wam" and variant["model_rotation_alpha"] == 1
    assert manifest["raw_checkpoint_reference"] is raw_flag and result["raw_checkpoint_reference"] is raw_flag
    assert manifest["reference_intervention"] == (
        "raw" if variant["model_rotation_alpha"] == 1 else "model_rotation_attenuation"
    )
    assert manifest["oracle_is_wam_success_evidence"] is False and result["oracle_is_wam_success_evidence"] is False
    assert manifest["inference_latency_excluded"] is True and result["inference_latency_excluded"] is True
    assert manifest["environment"]["CUDA_VISIBLE_DEVICES"] == manifest["environment"]["LD_LIBRARY_PATH"] == ""
    for path, signature in manifest["execution_sha256"].items():
        evidence.check(path, signature)
    required_runtime = [
        "src/s2a/stability_recovery_simulation.py",
        "src/s2a/stability_position_simulation.py",
        "scripts/run_wam_position_matrix.py",
        "src/s2a/gripper_position_hold.py",
        "src/s2a/gripper_intent_hold.py",
        "src/s2a/reference_governor_v2.py",
        "src/s2a/wam_regularization.py",
        "src/s2a/osc.py",
        "src/s2a/action_reference.py",
    ]
    assert all(str((REPO / p).resolve()) in manifest["execution_sha256"] for p in required_runtime)
    assert set(manifest["simulator_source_sha256"]) <= set(manifest["execution_sha256"])
    evidence.check(case / "model.xml", manifest["initial_model_xml_sha256"])
    template = evidence.json(config["policy_template"])
    h = np.asarray(evidence.json(template["tool_contract"])["source_from_tcp"])
    q = np.asarray(controller["source_from_control_rotation"])
    close(q.T @ q, np.eye(3))
    close(np.linalg.det(q), 1)
    assert chunks and len(chunks) == result["chunks"]
    assert chunks[0]["step"] == 0
    assert controller["actual_initial_physics_sha256"] == record["initial_state_sha256"]
    assert (
        hashlib.sha256(np.asarray(record["initial_state"], dtype=float).tobytes()).hexdigest()
        == record["initial_state_sha256"]
    )
    assert chunks[0]["physics_before_sha256"] == controller["actual_initial_physics_sha256"]
    close(controller["aligned_initial_motor_target"], rows[0]["motor_target_before"])
    initial = rows[0]["opening_before"]
    target = (0.079 * initial + 0.001) / 0.04 - 1
    close(controller["aligned_initial_motor_target"], [target, -target])
    dt = 1 / config["control_fps"]
    contract = controller["contract"]
    assert contract["gripper_model"] == "PandaGripper" and contract["control_frame"] == "MuJoCo world"
    assert contract["control_freq"] == 20 and contract["model_timestep"] == 0.002
    substeps = round(dt / contract["model_timestep"])
    assert substeps == 25
    close(substeps * contract["model_timestep"], dt)
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    arm = OSCStateToAction(
        interface,
        contract["kp"],
        contract["kd"],
        dt,
        compensate=True,
        inverse_weight=config["inverse_weight"],
        max_command_rate=config["max_command_rate"],
    )
    position = variant["gripper"] == "position_hold"
    gripper_type, config_type, config_name = (
        (GripperPositionHold, GripperPositionHoldConfig, "gripper_position_hold")
        if position
        else (GripperIntentHold, GripperIntentHoldConfig, "gripper_intent_hold")
    )
    gripper = gripper_type(initial, config=config_type(**config[config_name]))
    scalar_gripper = GripperReplay(initial, config[config_name], position)
    assert controller["gripper_config"] == asdict(gripper.config)
    assert asdict(gripper.config) == asdict(config_type())
    previous = np.zeros(7)
    references, native_source_outcome = None, None
    if source == "oracle":
        assert checkpoint is None and variant["model_rotation_alpha"] == 1
        assert not manifest["raw_checkpoint_reference"]
        assert manifest["oracle_reference_mode"] == "fixed_world_chunks_selected_by_current_physical_step"
        paths = [p for p in manifest["execution_sha256"] if p.endswith("/reference.npz")]
        assert len(paths) == 1
        with np.load(paths[0]) as data:
            references, oracle_opening = data["poses"] @ h, data["openings"]
        native_folder = Path(paths[0]).parent
        native_rows = [json.loads(line) for line in evidence.read(native_folder / "steps.jsonl").splitlines()]
        native_result = evidence.json(native_folder / "result.json")
        assert native_result["error"] is None and native_result["steps"] == len(native_rows) == 600
        native_source_outcome = task_outcome(
            [r["success"] for r in native_rows], expected_steps=600, stable_steps=config["stable_success_last_steps"]
        )
        assert native_source_outcome["stable"]
        close(references, np.asarray([r["after_source"] for r in native_rows]) @ h)
        close(oracle_opening, [r["opening_after"] for r in native_rows])
    else:
        ck = next(c for c in config["checkpoints"] if c["name"] == checkpoint)
        info = manifest["server_info"]
        expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
        assert Path(info["checkpoint_path"]).resolve() == expected.resolve() and not info["ema_weights"]
        assert Path(info["config_file"]).resolve() == (Path(ck["run"]) / "config.yaml").resolve()
        assert info["acceptance_intervention"]["active_slot_mask"] is True and ck["gpu"] in (4, 5, 6, 7)
        for relative in ("scripts/serve_wam_acceptance.py", "src/s2a/wam_runtime.py"):
            evidence.check(REPO / relative, info["acceptance_intervention"]["code"]["files_sha256"][relative])
    tracking = {
        name: []
        for name in (
            "admitted",
            "raw_at_model_clock",
            "effective_at_model_clock",
            "raw_at_nominal_clock",
            "effective_at_nominal_clock",
        )
    }
    zero_motor_steps, max_goal_rotation_error = 0, 0.0
    governor = RecoveryReplay(config["reference_governor"]) if full and variant["governor"] else None
    bridge_gates = []
    max_reference_speed = np.zeros(3)
    chunks_by_step = {c["step"]: (i, c) for i, c in enumerate(chunks)}
    assert len(chunks_by_step) == len(chunks)
    chunk, count = None, 0
    for step, row in enumerate(rows):
        assert row["step"] == step
        due = step == 0
        if step:
            due = (
                rows[step - 1]["reference_progress"]["complete"]
                if variant["governor"]
                else step - chunk["step"] >= count
            ) or step - chunk["step"] >= config["max_observation_interval_steps"]
        assert bool(due) == (step in chunks_by_step), "requery timing differs from causal completion/budget"
        close(row["actual_dt"], dt, tolerance=1e-9)
        close(row["physical_time_after_s"] - row["physical_time_before_s"], dt, tolerance=1e-9)
        if step:
            close(row["measured_source"], rows[step - 1]["after_source"])
            close(row["gripper_qpos_before"], rows[step - 1]["gripper_qpos_after"])
            close(row["opening_before"], rows[step - 1]["opening_after"])
            close(row["motor_opening_before"], rows[step - 1]["motor_opening_after"])
            close(row["motor_target_before"], rows[step - 1]["motor_target_after"])
            close(row["physical_time_before_s"], rows[step - 1]["physical_time_after_s"])
        measured, after, desired = (np.asarray(row[k]) for k in ("measured_source", "after_source", "desired_source"))
        desired_tcp = np.asarray(row["desired_tcp"])
        close(desired @ h, desired_tcp)
        if step in chunks_by_step:
            chunk_index, chunk = chunks_by_step[step]
            anchor = np.asarray(chunk["anchor_tcp"])
            close(anchor, measured @ h)
            close(chunk["anchor_opening"], row["opening_before"])
            assert chunk["physics_unchanged"]
            assert chunk["physics_before_sha256"] == chunk["physics_after_sha256"]
            close(chunk["physics_before_time_s"], row["physical_time_before_s"])
            close(chunk["physics_after_time_s"], row["physical_time_before_s"])
            raw, effective = np.asarray(chunk["raw128"]), np.asarray(chunk["effective128"])
            assert raw.ndim == 2 and raw.shape[1] == 128 and np.isfinite(raw).all() and not raw[:, 10:].any()
            assert effective.shape == raw.shape and np.isfinite(effective).all() and not effective[:, 10:].any()
            close(chunk["raw_targets_tcp"], decode(anchor, raw))
            close(chunk["effective_targets_tcp"], decode(anchor, effective))
            close(effective[:, :3], raw[:, :3])
            close(effective[:, 9:], raw[:, 9:])
            alpha = variant["model_rotation_alpha"]
            if alpha == 1:
                np.testing.assert_array_equal(raw, effective)
            else:
                # Per-delta SO(3), not cumulative orientation, is attenuated.
                for a, b in zip(raw, effective, strict=True):
                    ar = decode(np.eye(4), a[None])[0][:3, :3]
                    br = decode(np.eye(4), b[None])[0][:3, :3]
                    close(br, Rotation.from_rotvec(alpha * Rotation.from_matrix(ar).as_rotvec()).as_matrix())
            openings = np.clip(raw[:, 9], 0, 1)
            close(chunk["openings"], openings)
            count = chunk["execution_endpoint_count"]
            assert count == min(config["online_execution_horizon"], len(raw))
            raw_poses = np.asarray(chunk["raw_targets_tcp"])[:count]
            effective_poses = np.asarray(chunk["effective_targets_tcp"])[:count]
            if source == "oracle":
                close(chunk["raw_targets_tcp"], references[step : step + len(raw)])
                close(openings, oracle_opening[step : step + len(raw)])
                assert chunk["response_identity"] is None and chunk["request_sha256"] is None
            else:
                verify_query(config, case, chunk_index, chunk, evidence, checkpoint)
            if governor is not None:
                mapping(
                    chunk["retiming"],
                    governor.chunk(anchor, effective_poses, openings[:count], row["opening_before"], dt),
                )
        assert row["chunk_index"] == chunk_index and row["chunk_physical_step"] == step - chunk["step"]
        assert step - chunk["step"] < config["max_observation_interval_steps"]
        if not variant["governor"]:
            assert step - chunk["step"] < count
            close(desired_tcp, effective_poses[step - chunk["step"]])
            close(row["opening_target"], openings[step - chunk["step"]])
            assert row["reference_progress"] is None and not row["bridging"] and not row["bridge_recovery_advanced"]
            close(row["model_time_s"], (step - chunk["step"] + 1) * dt)
            close(row["retimed_clock_s"], row["model_time_s"])
        elif governor is not None:
            expected, bridge_gate = governor.step(measured @ h, dt)
            mapping(row["reference_progress"], expected)
            close(row["desired_tcp"], expected["target_tcp"])
            close(row["opening_target"], expected["opening"])
            for key in ("model_time_s", "retimed_clock_s", "bridging", "bridge_recovery_advanced"):
                close(row[key], expected[key])
            if bridge_gate is not None:
                bridge_gates.append(dict(step=step, **bridge_gate))
            previous_pose = anchor if step == 0 else np.asarray(rows[step - 1]["desired_tcp"])
            previous_opening = initial if step == 0 else rows[step - 1]["opening_target"]
            motion = error(desired_tcp, previous_pose)
            speeds = np.array(
                [
                    np.linalg.norm(motion[:3]) / dt,
                    np.linalg.norm(motion[3:]) / dt,
                    abs(row["opening_target"] - previous_opening) / dt,
                ]
            )
            max_reference_speed = np.maximum(max_reference_speed, speeds)
            limits = config["reference_governor"]
            assert np.all(
                speeds
                <= np.array(
                    [
                        limits["max_translation_speed_m_s"],
                        limits["max_rotation_speed_rad_s"],
                        limits["max_opening_speed_s"],
                    ]
                )
                + 1e-9
            )
            previous_clock = 0 if step == chunk["step"] else rows[step - 1]["model_time_s"]
            assert previous_clock - 1e-10 <= row["model_time_s"] <= count * dt + 1e-10
        raw_pose, raw_opening = sample(
            anchor, raw_poses, openings[:count], chunk["anchor_opening"], row["model_time_s"], dt
        )
        effective_pose, effective_opening = sample(
            anchor, effective_poses, openings[:count], chunk["anchor_opening"], row["model_time_s"], dt
        )
        close(row["raw_desired_tcp"], raw_pose)
        close(row["effective_desired_tcp"], effective_pose)
        close(row["raw_opening_at_model_time"], raw_opening)
        close(row["effective_opening_at_model_time"], effective_opening)
        nominal_time = (step - chunk["step"] + 1) * dt
        nominal_raw, _ = sample(anchor, raw_poses, openings[:count], chunk["anchor_opening"], nominal_time, dt)
        nominal_effective, _ = sample(
            anchor, effective_poses, openings[:count], chunk["anchor_opening"], nominal_time, dt
        )
        for key, target_pose in [
            ("admitted", desired_tcp),
            ("raw_at_model_clock", raw_pose),
            ("effective_at_model_clock", effective_pose),
            ("raw_at_nominal_clock", nominal_raw),
            ("effective_at_nominal_clock", nominal_effective),
        ]:
            tracking[key].append(error(target_pose, after @ h))
        close(row["tcp_error"], tracking["admitted"][-1])
        close(row["source_error"], error(desired, after))
        close(row["raw_model_tcp_error"], tracking["raw_at_model_clock"][-1])
        close(row["effective_model_tcp_error"], tracking["effective_at_model_clock"][-1])
        reconstructed = arm.convert(measured, desired, row["measured_twist"])
        close(row["command"][:6], reconstructed["command"])
        assert row["saturated"] == reconstructed["saturated"] and row["rate_limited"] == reconstructed["rate_limited"]
        motor_before, motor_after = np.asarray(row["motor_target_before"]), np.asarray(row["motor_target_after"])
        motor_width = float(np.clip((0.04 * (motor_before[0] + 1) - 0.001) / 0.079, 0, 1))
        close(row["motor_opening_before"], motor_width)
        for label in ("before", "after"):
            qpos = row[f"gripper_qpos_{label}"]
            close(row[f"opening_{label}"], np.clip((qpos[0] - qpos[1] - 0.001) / 0.079, 0, 1))
        motor_after_width = float(np.clip((0.04 + 0.02 * (motor_after[0] - motor_after[1]) - 0.001) / 0.079, 0, 1))
        close(row["motor_opening_after"], motor_after_width)
        grip = gripper.update(
            row["opening_target"],
            row["opening_before"],
            motor_opening=motor_width,
            previous_command=previous[-1],
            dt=dt,
        )
        mapping(row["gripper"], asdict(grip))
        scalar_command = scalar_gripper.step(
            row["opening_target"],
            row["opening_before"],
            motor_width,
            previous[-1],
            dt,
        )
        assert row["command"][-1] == grip.command == scalar_command
        assert row["gripper"]["closing_episode"] == scalar_gripper.latched
        command = np.asarray(row["command"])
        assert command[-1] in (-1, 0, 1)
        assert np.all(command[:6] >= interface.input_min - 1e-12)
        assert np.all(command[:6] <= interface.input_max + 1e-12)
        assert np.max(np.abs(command[:6] - previous[:6])) <= config["max_command_rate"] * dt + 1e-12
        close(motor_after, np.clip(motor_before + np.array([-1, 1]) * 0.01 * substeps * np.sign(command[-1]), -1, 1))
        close(row["actual_gripper_actuator_control"], [0.02, -0.02] + 0.02 * motor_after)
        if command[-1] == 0:
            np.testing.assert_array_equal(motor_before, motor_after)
            zero_motor_steps += 1
        delta = interface.decode(command[:6])
        close(row["controller_goal_position_after"], measured[:3, 3] + delta[:3])
        expected_goal = native_delta_rotation(delta[3:]) @ measured[:3, :3] @ q
        if not np.any(delta[3:]):
            expected_goal = np.asarray(row["controller_goal_before"])
        max_goal_rotation_error = max(
            max_goal_rotation_error, float(np.max(np.abs(np.asarray(row["controller_goal_after"]) - expected_goal)))
        )
        # Native quat2mat deliberately uses float32; the actual controller
        # SITE matrix is not logged per endpoint. A uniform declared numeric
        # tolerance covers that vendor conversion, not a task-success gate.
        close(row["controller_goal_after"], expected_goal, tolerance=1e-6)
        close(row["controller_goal_source_rotation_before"], np.asarray(row["controller_goal_before"]) @ q.T)
        previous = command
    outcome = task_outcome(
        [r["success"] for r in rows], expected_steps=600, stable_steps=config["stable_success_last_steps"]
    )
    assert result["task_outcome"] == outcome
    assert result["stable_success"] == outcome["stable"] and result["ever_success"] == outcome["ever"]
    target = TARGETS[record["task_id"]]
    for name in rows[0]["objects"]:
        first = rows[0]["objects"][name]
        initial_height = first["position"][2] - first["height_gain_m"]
        for row in rows:
            obj = row["objects"][name]
            assert type(obj["grasp_contact"]) is bool
            close(obj["height_gain_m"], obj["position"][2] - initial_height)
    admitted = rmse(tracking["admitted"])
    close(result["tcp_position_rmse_m"], admitted["position_rmse_m"])
    close(result["rotation_rmse_rad"], admitted["so3_rmse_rad"])
    consumption = []
    for i, chunk in enumerate(chunks):
        final_step = chunks[i + 1]["step"] - 1 if i + 1 < len(chunks) else len(rows) - 1
        consumed = rows[final_step]["model_time_s"]
        available = chunk["execution_endpoint_count"] * dt
        generated = len(chunk["raw128"]) * dt
        consumption.append(
            dict(
                chunk=i,
                first_step=chunk["step"],
                final_step=final_step,
                physical_steps=final_step - chunk["step"] + 1,
                generated_model_s=generated,
                execution_prefix_model_s=available,
                consumed_model_s=consumed,
                prefix_fraction=min(consumed / available, 1.0),
                discarded_on_requery_s=max(0.0, generated - consumed) if i + 1 < len(chunks) else 0.0,
                remaining_at_budget_s=max(0.0, generated - consumed) if i + 1 == len(chunks) else 0.0,
            )
        )
    model_consumed = sum(c["consumed_model_s"] for c in consumption)
    return dict(
        case=str(case),
        source=source,
        checkpoint=checkpoint,
        variant=variant["name"],
        record=record["name"],
        status="verified_basic" if not full else "verified",
        steps=600,
        chunks=len(chunks),
        oracle_native_source_outcome=native_source_outcome,
        independent_scalar_gripper_replay="passed",
        first_query003_regression=verify_first_query006(config, record, variant, case, chunks[0], evidence, checkpoint)
        if source == "wam"
        else None,
        outcome=outcome,
        zero_motor_steps=zero_motor_steps,
        controller_goal_rotation_max_abs_reconstruction_error=max_goal_rotation_error,
        controller_goal_rotation_tolerance=1e-6,
        tracking={k: rmse(v) for k, v in tracking.items()},
        target_grasp_contact_steps=sum(r["objects"][target]["grasp_contact"] for r in rows),
        object_metrics={
            name: dict(
                contact_steps=sum(r["objects"][name]["grasp_contact"] for r in rows),
                max_height_gain_m=max(r["objects"][name]["height_gain_m"] for r in rows),
            )
            for name in rows[0]["objects"]
        },
        independent_governor_replay=("passed" if governor else "pending") if variant["governor"] else "not_applicable",
        governor_diagnostics=dict(
            chunk_consumption=consumption,
            consumed_model_s=model_consumed,
            generated_model_s=sum(c["generated_model_s"] for c in consumption),
            model_consumed_per_physical_second=model_consumed / (len(rows) * dt),
            generated_reference_fraction=model_consumed / sum(c["generated_model_s"] for c in consumption),
            discarded_on_requery_s=sum(c["discarded_on_requery_s"] for c in consumption),
            remaining_at_budget_s=sum(c["remaining_at_budget_s"] for c in consumption),
            bridging_endpoint_count=sum(r["bridging"] for r in rows),
            bridge_recovery_advanced_count=sum(r["bridge_recovery_advanced"] for r in rows),
            paused_clock_endpoints=sum(
                r["reference_progress"] is not None and r["reference_progress"]["tracking_scale"] == 0 for r in rows
            ),
            nominal_tracking_paused_endpoints=sum(
                r["reference_progress"] is not None and r["reference_progress"]["nominal_tracking_scale"] == 0
                for r in rows
            ),
            max_admitted_reference_speeds=max_reference_speed.tolist() if governor else None,
            independently_recomputed_bridge_gates=bridge_gates,
        ),
        component_ablation=variant["model_rotation_alpha"] == 0,
        raw_checkpoint_evidence=source == "wam" and variant["model_rotation_alpha"] == 1,
    )


def fork_proof(evidence):
    original = evidence.read(BASE).decode()
    own = evidence.read(__file__).decode()

    def function(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "verify_case")
        return "".join(text.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])

    first, second = function(original), function(own)
    assert first.count("verify_first_query003(") == 1
    assert first.replace("verify_first_query003(", "verify_first_query006(") == second
    return dict(
        original_commit="140f7e8",
        original_file=BASE,
        original_file_sha256=BASE_SHA,
        original_verify_case_sha256=hashlib.sha256(first.encode()).hexdigest(),
        fork_verify_case_sha256=hashlib.sha256(second.encode()).hexdigest(),
        only_first_query_source_callback_changed=True,
        all_other_step_checks_literal_identical=True,
        unified_diff="".join(
            difflib.unified_diff(
                first.splitlines(keepends=True),
                second.splitlines(keepends=True),
                fromfile="140f7e8:verify_case",
                tofile="006:verify_case",
            )
        ),
        dynamic_monkeypatch_used=False,
    )


def resources(config, launch, evidence):
    directory = Path(config["output_dir"]) / "orchestration" / ("8400_masked_" + "_".join(VARIANTS))
    manifest = evidence.json(directory / "manifest.json")
    assert manifest["config"] == config and manifest["physical_gpu"] == launch["physical_gpu"] == 4
    assert manifest["coordinator_pid"] == launch["coordinator_pid"]
    evidence.check(CONFIG, manifest["config_sha256"])
    expected = [
        [
            name,
            [
                str(REPO / ".venv/bin/python"),
                "scripts/run_wam_position_matrix.py",
                "--config",
                CONFIG,
                "--seed",
                "42",
                "--source",
                "wam",
                "--variant",
                name,
                "--checkpoint",
                "8400_masked",
            ],
        ]
        for name in VARIANTS
    ]
    assert manifest["jobs"] == expected
    fingerprints_path = "outputs/wam_independent_20261020/checkpoint_fingerprints.json"
    evidence.check(fingerprints_path, manifest["fingerprint_manifest_sha256"])
    weights = next(item for item in evidence.json(fingerprints_path)["checkpoints"] if item["name"] == "8400_masked")
    ck = config["checkpoints"][0]
    assert Path(weights["checkpoint"]) == Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
    for filename, signature in weights["files"].items():
        stat = Path(filename).stat()
        assert stat.st_size == signature["bytes"] and stat.st_mtime_ns == signature["mtime_ns"]
    model = evidence.json(directory / "model_process.json")
    assert model["gpu"] == 4 and model["port"] == 9344
    processes = dict(
        coordinator=process_environment(launch["coordinator_pid"], "", start_ticks=launch["coordinator_start_ticks"]),
        model=process_environment(model["pid"], "4"),
        matrix_workers={},
    )
    for variant, command in expected:
        path = directory / f"{variant}_process.json"
        if path.exists():
            worker = evidence.json(path)
            assert worker["command"] == command
            processes["matrix_workers"][variant] = process_environment(worker["pid"], "")
    info_path = directory / "server_info.json"
    if info_path.exists():
        info = evidence.json(info_path)
        assert Path(info["checkpoint_path"]).resolve() == Path(weights["checkpoint"]).resolve()
        assert not info["ema_weights"] and info["acceptance_intervention"]["active_slot_mask"]
    error_path = directory / "error.json"
    recorded_error = evidence.json(error_path) if error_path.exists() else None
    released = False
    if (directory / "complete.json").exists() and (directory / "release.json").exists():
        assert evidence.json(directory / "complete.json") == dict(complete=True, variants=list(VARIANTS))
        release = evidence.json(directory / "release.json")
        assert release["model_pid"] == model["pid"] and release["model_returncode"] in (0, -15)
        assert set(processes["matrix_workers"]) == set(VARIANTS)
        released = not processes["model"]["present"] and not any(
            p["present"] for p in processes["matrix_workers"].values()
        )
    return dict(
        processes=processes,
        recorded_execution_error=recorded_error,
        owned_model_and_matrix_workers_released_normally=released and recorded_error is None,
        reused_complete_weight_fingerprint_files=len(weights["files"]),
        full_weight_bytes_rehashed_now=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-verified", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("006 audit requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO / "outputs"):
        raise ValueError("audit snapshot must be a new repository outputs file")
    evidence, reusable, cases = Evidence(), {}, []
    evidence.check(BASE, BASE_SHA)
    assert hashlib.sha256(subprocess.check_output(["git", "show", f"140f7e8:{BASE}"])).hexdigest() == BASE_SHA
    evidence.read("scripts/audit_wam_earlier_recovery.py")
    proof = fork_proof(evidence)
    config = evidence.json(CONFIG)
    frozen = subprocess.check_output(["git", "show", f"6694832:{CONFIG}"])
    evidence.check(CONFIG, hashlib.sha256(frozen).hexdigest())
    reference = evidence.json("configs/wam_position_pilot_8800_g10_20261021.json")
    omitted = {"checkpoints", "output_dir", "notes"}
    assert {k: v for k, v in config.items() if k not in omitted} == {
        k: v for k, v in reference.items() if k not in omitted
    }
    assert config["records"] == ["task0_init0", "task1_init0", "task2_init0"] and config["sampling_seeds"] == [42]
    assert config["online_max_steps"] == 600 and tuple(v["name"] for v in config["variants"]) == VARIANTS
    assert len(config["checkpoints"]) == 1
    ck = config["checkpoints"][0]
    assert (ck["name"], ck["iteration"], ck["gpu"], ck["port"]) == ("8400_masked", "iter_000008400", 4, 9344)
    launch_path = "outputs/wam_stability_20261021/position_earlier_coordinator/launch.json"
    launch = evidence.json(launch_path)
    evidence.check(CONFIG, launch["config_sha256"])
    assert launch["new_trials"] == 12 and launch["max_physics_steps"] == 7200 and launch["max_h40_model_queries"] == 180
    assert launch["budget_git_commit"] == "6694832"
    gate_path = "outputs/wam_stability_20261021/position_earlier_coordinator/qualification_and_release_gate.json"
    evidence.check(gate_path, launch["qualification_gate_sha256"])
    gate = evidence.json(gate_path)
    assert gate["gate_passed"] and gate["old12_complete_and_released"] and gate["physical_gpu"] == 4
    for filename, signature in gate["rechecked_input_sha256"].items():
        evidence.check(filename, signature)
    qualification = verify_004_gate(evidence)
    assert qualification["qualified"] and qualification["oracle_cases"] == 9
    if args.reuse_verified:
        previous_path = args.reuse_verified.resolve()
        assert previous_path.is_relative_to(REPO / "outputs")
        previous = evidence.json(previous_path)
        assert previous["schema_version"] == SCHEMA
        own_path = str(Path(__file__).resolve())
        assert previous["source_before_after_hashes"][own_path] == evidence.hashes[own_path]
        for filename, signature in previous["source_before_after_hashes"].items():
            evidence.check(filename, signature)
        reusable = {str(Path(c["case"]).resolve()): c for c in previous["cases"] if c["status"] == "verified"}
    evidence.check(config["fixture"], config["fixture_sha256"])
    fixtures = {r["name"]: r for r in evidence.json(config["fixture"])["records"]}
    observed_requests = 0
    for variant in config["variants"]:
        for name in config["records"]:
            case = Path(config["output_dir"]) / "simulation" / f"8400_masked_{variant['name']}" / f"{name}_s42"
            requests = len(list(case.glob("request_*.json")))
            assert requests <= 15
            observed_requests += requests
            identity = dict(
                case=str(case), source="wam", checkpoint="8400_masked", variant=variant["name"], record=name
            )
            marker = case / "result.json"
            if not marker.exists():
                cases.append(dict(identity, status="partial" if case.exists() else "missing"))
                continue
            try:
                json.loads(marker.read_bytes())
            except json.JSONDecodeError:
                cases.append(dict(identity, status="partial", reason="result marker being written"))
                continue
            result = evidence.json(marker)
            if result["error"]:
                cases.append(dict(identity, status="execution_error", error=result["error"]))
                continue
            assert requests == result["chunks"]
            cached = reusable.get(str(case.resolve()))
            if cached is not None:
                cases.append(dict(cached, reused_from=str(args.reuse_verified.resolve())))
                continue
            try:
                cases.append(
                    verify_case(
                        config,
                        fixtures[name],
                        variant,
                        case,
                        evidence,
                        source="wam",
                        checkpoint="8400_masked",
                        full=True,
                    )
                )
            except Exception as exc:
                cases.append(dict(identity, status="audit_error", error=repr(exc), traceback=traceback.format_exc()))
    assert len(cases) == 12 and observed_requests <= 180
    resource_snapshot = resources(config, launch, evidence)
    groups = []
    for variant in config["variants"]:
        verified = [c for c in cases if c["variant"] == variant["name"] and c["status"] == "verified"]
        complete = len(verified) == 3
        groups.append(
            dict(
                variant=variant["name"],
                model_rotation_alpha=variant["model_rotation_alpha"],
                expected_cases=3,
                verified_cases=len(verified),
                complete=complete,
                stable_successes_among_verified=sum(c["outcome"]["stable"] for c in verified),
                ever_successes_among_verified=sum(c["outcome"]["ever"] for c in verified),
                stable_rate=sum(c["outcome"]["stable"] for c in verified) / 3 if complete else None,
                raw_checkpoint_reference=variant["model_rotation_alpha"] == 1,
                component_ablation_only=variant["model_rotation_alpha"] == 0,
                tracking_pooled_equal_length_cases={
                    ref: {
                        metric: float(np.sqrt(np.mean([c["tracking"][ref][metric] ** 2 for c in verified])))
                        for metric in ("position_rmse_m", "so3_rmse_rad")
                    }
                    for ref in verified[0]["tracking"]
                }
                if complete
                else None,
            )
        )
    all_verified = all(c["status"] == "verified" for c in cases)
    evidence.recheck()
    value = dict(
        schema_version=SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="only12_frozen006_8400_position_wam_development_trials",
        complete=all_verified and resource_snapshot["owned_model_and_matrix_workers_released_normally"],
        all12_cases_independently_verified=all_verified,
        reused_verified_cases=sum("reused_from" in c for c in cases),
        literal_fork_verification=proof,
        registered_budget=dict(wam_trials=12, physics_steps=7200, model_queries_upper_bound=180),
        verified_physics_steps=sum(c.get("steps", 0) for c in cases if c["status"] == "verified"),
        observed_request_files=observed_requests,
        cases=cases,
        groups=groups,
        resources=resource_snapshot,
        existing004_qualification_reference=dict(
            qualification,
            added_oracle_trials=0,
            qualification_physics_reexecuted=0,
            oracle_excluded_from_wam_denominator=True,
        ),
        source_before_after_hashes=evidence.hashes,
        warnings=[
            "Only12 old-development WAM trials are traversed; nine004 oracle qualifications are referenced, never rerun or counted in WAM denominator.",
            "Every140f7e8 verify_case check is literal unchanged except the first-query source callback, authenticated by source diff; no dynamic patch.",
            "first_query003_regression key is retained from the literal fork, but its006 source is the corresponding8400 old12 recovery case.",
            "Alpha.5 is a reference intervention; alpha0 is constant-orientation component diagnosis, not raw full-checkpoint performance.",
            "Native predicates/contact/object state are metrics only; no new holdout, real hardware or pretraining-unseen claim.",
            "Partial/error cases or pending owned resource release prevent completion; no partial winner is selected.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                output=str(output),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                complete=value["complete"],
                statuses={s: sum(c["status"] == s for c in cases) for s in {c["status"] for c in cases}},
            )
        )
    )


if __name__ == "__main__":
    main()
