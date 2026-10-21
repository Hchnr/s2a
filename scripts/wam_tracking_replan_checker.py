"""Independent016 saved-log checker: literal014 mechanics plus causal timing.

No model or simulator is launched. The production timing policy is not imported.
All014 authentication, geometry, command, gripper, clock and native outcome
checks are retained. Only the new source/schema fields and query timing differ.
"""

import ast
import difflib
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_wam_position import (
    REPO,
    TARGETS,
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
)
from wam_position_acceptance_checker import source_proof as original_source_proof
from wam_position_acceptance_checker import verify_query

BASE = "scripts/wam_position_acceptance_checker.py"
BASE_SHA = "7a90309b1b2ea2b75429f1f0cecb4e5036aab43281cb66c4214a57de4df2d85f"
CONTRACT = dict(
    min_interval_steps=8, max_interval_steps=40, position_limit_m=0.04, rotation_limit_rad=0.4, consecutive_steps=2
)


def validate_contract(config, variant):
    assert config["tracking_replan"] == CONTRACT
    assert config["max_observation_interval_steps"] == config["online_execution_horizon"] == 40
    assert variant == dict(
        name="position_hold", rotation="response", gripper="position_hold", governor=False, model_rotation_alpha=1.0
    )


def expected_decision(step, last_query_step, endpoint_count, violations):
    """Pre-step decision uses only the previous completed endpoint counter."""
    age = 0 if last_query_step is None else step - last_query_step
    if last_query_step is None:
        reason = "initial"
    elif age >= endpoint_count:
        reason = "reference_exhausted"
    elif age >= 40:
        reason = "maximum_interval"
    elif age >= 8 and violations >= 2:
        reason = "tracking_limit"
    else:
        reason = None
    return dict(
        physical_step=step,
        age_steps=age,
        previous_violation_count=violations,
        query_due=reason is not None,
        query_reason=reason,
    )


def expected_endpoint(tcp_error, previous_violations):
    """Independently use reconstructed after-TCP errors, never saved booleans."""
    error_vector = np.asarray(tcp_error, dtype=float)
    assert error_vector.shape == (6,) and np.isfinite(error_vector).all()
    position = float(np.sqrt(np.dot(error_vector[:3], error_vector[:3])))
    rotation = float(np.sqrt(np.dot(error_vector[3:], error_vector[3:])))
    exceeded = position > 0.04 or rotation > 0.4
    violations = previous_violations + 1 if exceeded else 0
    return dict(
        position_error_m=position,
        rotation_error_rad=rotation,
        exceeded=exceeded,
        consecutive_violations_after_step=violations,
    ), violations


ALLOWED_SOURCE_CHANGES = [
    (
        '    result = evidence.json(case / "result.json")',
        '    validate_contract(config, variant)\n    result = evidence.json(case / "result.json")',
    ),
    (
        '    assert manifest["source"] == result["source"] == source',
        '    assert manifest["source"] == result["source"] == source\n    assert manifest["schema_version"] == "s2a.tracking-replan-execution.v1"\n    assert manifest["tracking_replan"] == config["tracking_replan"]',
    ),
    (
        '        "src/s2a/stability_recovery_simulation.py",',
        '        "src/s2a/tracking_replan.py",\n        "src/s2a/stability_tracking_simulation.py",\n        "scripts/run_wam_tracking_matrix.py",\n        "src/s2a/stability_recovery_simulation.py",',
    ),
    ("    chunk, count = None, 0", "    chunk, count = None, 0\n    violations, decisions = 0, []"),
    (
        '        due = step == 0\n        if step:\n            due = (\n                rows[step - 1]["reference_progress"]["complete"]\n                if variant["governor"]\n                else step - chunk["step"] >= count\n            ) or step - chunk["step"] >= config["max_observation_interval_steps"]\n        assert bool(due) == (step in chunks_by_step), "requery timing differs from causal completion/budget"',
        '        decision = expected_decision(step, None if chunk is None else chunk["step"], count, violations)\n        mapping(row["query_decision"], decision)\n        assert decision["query_due"] == (step in chunks_by_step), "requery differs from completed TCP evidence"\n        if decision["query_due"]:\n            mapping(chunks_by_step[step][1]["query_decision"], decision)\n            decisions.append(decision)\n            violations = 0',
    ),
    (
        '        close(row["tcp_error"], tracking["admitted"][-1])',
        '        close(row["tcp_error"], tracking["admitted"][-1])\n        endpoint, violations = expected_endpoint(tracking["admitted"][-1], violations)\n        mapping(row["tracking_replan"], endpoint)',
    ),
    (
        "        zero_motor_steps=zero_motor_steps,",
        '        zero_motor_steps=zero_motor_steps,\n        tracking_replan_replay=dict(\n            independent_causal_replay="passed",\n            all600_endpoints_verified=True,\n            query_decisions=decisions,\n            query_reason_counts={reason: sum(d["query_reason"] == reason for d in decisions) for reason in (\n                "initial", "reference_exhausted", "maximum_interval", "tracking_limit"\n            )},\n        ),',
    ),
]


def source_proof(evidence):
    upstream = original_source_proof(evidence)
    evidence.check(BASE, BASE_SHA)
    base, own = evidence.read(BASE).decode(), evidence.read(__file__).decode()

    def function(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "verify_case")
        return "".join(text.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])

    original = function(base)
    expected = original
    for before, after in ALLOWED_SOURCE_CHANGES:
        assert expected.count(before) == 1
        expected = expected.replace(before, after)
    actual = function(own)
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(actual))
    return dict(
        base_file=BASE,
        base_sha256=BASE_SHA,
        upstream014_proof=upstream,
        new_checker_sha256=hashlib.sha256(own.encode()).hexdigest(),
        other014_case_ast_nodes_exact=True,
        no_production_tracking_policy_import=True,
        no_dynamic_patch_or_generated_function=True,
        source_unified_diff="".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile="014:verify_case",
                tofile="016:verify_case",
            )
        ),
    )


def verify_case(
    config, record, variant, case, evidence, *, source, checkpoint, full, sampling_seed, first_query_validator
):
    assert type(sampling_seed) is int and sampling_seed in config["sampling_seeds"]
    assert callable(first_query_validator) if source == "wam" else first_query_validator is None
    validate_contract(config, variant)
    result = evidence.json(case / "result.json")
    manifest = evidence.json(case / "manifest.json")
    controller = evidence.json(case / "controller.json")
    chunks = evidence.json(case / "chunks.json")
    rows = [json.loads(line) for line in evidence.read(case / "steps.jsonl").splitlines()]
    assert len(rows) == result["steps"] == result["expected_steps"] == config["online_max_steps"] == 600
    assert result["error"] is None and result["complete"] and result["execution_inputs_unchanged"]
    assert manifest["source"] == result["source"] == source
    assert manifest["schema_version"] == "s2a.tracking-replan-execution.v1"
    assert manifest["tracking_replan"] == config["tracking_replan"]
    assert manifest["experiment"] == config and manifest["variant"] == variant and manifest["record"] == record
    assert result["sampling_seed"] == sampling_seed and result["variant"] == variant["name"]
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
        "src/s2a/tracking_replan.py",
        "src/s2a/stability_tracking_simulation.py",
        "scripts/run_wam_tracking_matrix.py",
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
    violations, decisions = 0, []
    for step, row in enumerate(rows):
        assert row["step"] == step
        decision = expected_decision(step, None if chunk is None else chunk["step"], count, violations)
        mapping(row["query_decision"], decision)
        assert decision["query_due"] == (step in chunks_by_step), "requery differs from completed TCP evidence"
        if decision["query_due"]:
            mapping(chunks_by_step[step][1]["query_decision"], decision)
            decisions.append(decision)
            violations = 0
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
                verify_query(config, case, chunk_index, chunk, evidence, checkpoint, sampling_seed=sampling_seed)
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
        endpoint, violations = expected_endpoint(tracking["admitted"][-1], violations)
        mapping(row["tracking_replan"], endpoint)
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
        first_query003_regression=first_query_validator(
            config, record, variant, case, chunks[0], evidence, checkpoint, sampling_seed=sampling_seed
        )
        if source == "wam"
        else None,
        outcome=outcome,
        zero_motor_steps=zero_motor_steps,
        tracking_replan_replay=dict(
            independent_causal_replay="passed",
            all600_endpoints_verified=True,
            query_decisions=decisions,
            query_reason_counts={
                reason: sum(d["query_reason"] == reason for d in decisions)
                for reason in ("initial", "reference_exhausted", "maximum_interval", "tracking_limit")
            },
        ),
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
