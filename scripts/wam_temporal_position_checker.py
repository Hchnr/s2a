"""Independent020 checker: literal014 mechanics with causal endpoint ensembles.

No production temporal module is imported. Complete original128 outputs and
latest-query diagnostics retain014 checks; admitted targets alone are rebuilt
from already-generated, fixed-world predictions for the same integer endpoint.
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
from PIL import Image
from wam_position_acceptance_checker import source_proof as original_source_proof

BASE = "scripts/wam_position_acceptance_checker.py"
BASE_SHA = "7a90309b1b2ea2b75429f1f0cecb4e5036aab43281cb66c4214a57de4df2d85f"
CONTRACT = dict(
    horizon_steps=40,
    query_interval_steps=8,
    identity_atol=1e-12,
    minimum_singular_value=1e-6,
    method="uniform_chordal_world_same_endpoint",
)


def validate_contract(config, variant):
    assert config["reference_ensemble"] == CONTRACT
    assert config["online_chunk_size"] == config["online_execution_horizon"] == 40
    assert config["max_observation_interval_steps"] == 8 and config["control_fps"] == 20
    assert variant == dict(
        name="temporal_equal", rotation="response", gripper="position_hold", governor=False, model_rotation_alpha=1.0
    )


def reconstruct_ensemble(physical_step, chunks):
    """Independent stateless integer-time reconstruction, without future inputs.

    q+j+1=k+1 implies j=k-q. Each retained query is integrated at its saved
    observed world anchor; it is never rebased at a newer measured pose.
    """
    assert type(physical_step) is int and physical_step >= 0
    available = [chunk for chunk in chunks if chunk["step"] <= physical_step]
    assert [chunk["step"] for chunk in available] == list(range(0, physical_step + 1, 8))
    contributors = []
    for chunk in available:
        q = chunk["step"]
        assert type(q) is int and q >= 0
        j = physical_step - q
        if j >= len(chunk["raw128"]):
            continue
        raw = np.asarray(chunk["raw128"], dtype=float)
        assert raw.ndim == 2 and raw.shape[1] == 128 and 1 <= len(raw) <= 40
        assert np.isfinite(raw).all() and not raw[:, 10:].any()
        np.testing.assert_array_equal(raw, chunk["effective128"])
        world = decode(np.asarray(chunk["anchor_tcp"]), raw)
        pose, opening = world[j], float(np.clip(raw[j, 9], 0, 1))
        contributors.append(
            dict(query_step=q, endpoint_index=j, absolute_endpoint=physical_step + 1, target_tcp=pose, opening=opening)
        )
    assert 1 <= len(contributors) <= 5 and contributors[-1]["query_step"] == physical_step // 8 * 8
    count = len(contributors)
    poses = np.asarray([item["target_tcp"] for item in contributors])
    openings = np.asarray([item["opening"] for item in contributors])
    newest, newest_opening = poses[-1], float(openings[-1])
    identity = bool(np.max(np.abs(poses - newest)) <= 1e-12 and np.max(np.abs(openings - newest_opening)) <= 1e-12)
    determinant, minimum = None, None
    if identity:
        target, opening = newest.copy(), newest_opening
    else:
        # This is the registered chordal/polar mean, not an axis-angle mean.
        mean = sum((pose[:3, :3] for pose in poses), np.zeros((3, 3))) / count
        u, singular, vt = np.linalg.svd(mean)
        determinant, minimum = float(np.linalg.det(mean)), float(min(singular))
        assert np.isfinite(mean).all() and determinant > 0 and minimum >= 1e-6
        target = np.eye(4)
        target[:3, 3] = sum((pose[:3, 3] for pose in poses), np.zeros(3)) / count
        target[:3, :3] = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        opening = float(sum(openings) / count)
        close(target[:3, :3].T @ target[:3, :3], np.eye(3))
        close(np.linalg.det(target[:3, :3]), 1)
    for item in contributors:
        item["weight"] = 1 / count
    expected = dict(
        physical_step=physical_step,
        absolute_endpoint=physical_step + 1,
        contributor_count=count,
        equal_weight=1 / count,
        newest_query_step=contributors[-1]["query_step"],
        identity_fast_path=identity,
        mean_rotation_determinant=determinant,
        minimum_singular_value=minimum,
        contributors=contributors,
    )
    return target, opening, expected


def compare_ensemble_log(saved, expected):
    """Integer chronology and equal weights are exact, not tolerance gates."""
    assert set(saved) == set(expected)
    for field in ("physical_step", "absolute_endpoint", "contributor_count", "newest_query_step"):
        assert type(saved[field]) is int and saved[field] == expected[field]
    assert type(saved["identity_fast_path"]) is bool
    assert saved["equal_weight"] == expected["equal_weight"]
    assert len(saved["contributors"]) == len(expected["contributors"])
    for actual, desired in zip(saved["contributors"], expected["contributors"], strict=True):
        for field in ("query_step", "endpoint_index", "absolute_endpoint"):
            assert type(actual[field]) is int and actual[field] == desired[field]
        assert actual["weight"] == desired["weight"]
        mapping(actual, desired)
    mapping(
        {key: value for key, value in saved.items() if key != "contributors"},
        {key: value for key, value in expected.items() if key != "contributors"},
    )


def source_proof(evidence):
    """All014 query/physics/control AST remains except explicitly listed edits."""
    parent = original_source_proof(evidence)
    evidence.check(BASE, BASE_SHA)
    base, own = evidence.read(BASE).decode(), evidence.read(__file__).decode()

    def function(text, name):
        node = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name)
        return "".join(text.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])

    expected = function(base, "verify_case")
    for old, new in ALLOWED_SOURCE_CHANGES:
        assert expected.count(old) == 1, old
        expected = expected.replace(old, new)
    actual = function(own, "verify_case")
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(actual))
    assert ast.dump(ast.parse(function(base, "verify_query"))) == ast.dump(ast.parse(function(own, "verify_query")))
    return dict(
        original014_source_proof=parent,
        original014_file=BASE,
        original014_sha256=BASE_SHA,
        new_checker_sha256=hashlib.sha256(own.encode()).hexdigest(),
        allowed_changes=[
            "fixed020 contract/schema/source guards",
            "reference-intervention identification; never raw checkpoint score",
            "independent same-absolute-endpoint admitted target/opening and contributor provenance",
            "additional ensemble replay result only",
        ],
        literal_query_ast_exact=True,
        all_other_case_ast_nodes_exact=True,
        production_temporal_module_imported=False,
        dynamic_patch_or_generated_function_used=False,
        source_unified_diff="".join(
            difflib.unified_diff(
                function(base, "verify_case").splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile="014:verify_case",
                tofile="020:verify_case",
            )
        ),
    )


ALLOWED_SOURCE_CHANGES = [
    (
        '    result = evidence.json(case / "result.json")',
        '    validate_contract(config, variant)\n    result = evidence.json(case / "result.json")',
    ),
    (
        '    assert manifest["source"] == result["source"] == source',
        '    assert manifest["source"] == result["source"] == source\n    assert manifest["schema_version"] == "s2a.temporal-equal-execution.v1"\n    assert manifest["reference_ensemble"] == config["reference_ensemble"]',
    ),
    ('    raw_flag = source == "wam" and variant["model_rotation_alpha"] == 1', "    raw_flag = False"),
    (
        '    assert manifest["reference_intervention"] == (\n        "raw" if variant["model_rotation_alpha"] == 1 else "model_rotation_attenuation"\n    )',
        '    assert manifest["reference_intervention"] == result["reference_intervention"] == "causal_temporal_equal_ensemble"',
    ),
    (
        '        "src/s2a/stability_recovery_simulation.py",',
        '        "src/s2a/temporal_ensemble.py",\n        "src/s2a/temporal_ensemble_simulation.py",\n        "scripts/run_wam_temporal_matrix.py",\n        "src/s2a/stability_recovery_simulation.py",',
    ),
    (
        '            close(desired_tcp, effective_poses[step - chunk["step"]])\n            close(row["opening_target"], openings[step - chunk["step"]])',
        '            ensemble_pose, ensemble_opening, ensemble_provenance = reconstruct_ensemble(step, chunks)\n            close(desired_tcp, ensemble_pose)\n            close(row["opening_target"], ensemble_opening)\n            compare_ensemble_log(row["temporal_ensemble"], ensemble_provenance)',
    ),
    (
        '        raw_checkpoint_evidence=source == "wam" and variant["model_rotation_alpha"] == 1,',
        '        raw_checkpoint_evidence=False,\n        reference_intervention="causal_temporal_equal_ensemble",\n        independent_temporal_ensemble_replay=dict(\n            passed=True, verified_endpoints=600, same_absolute_endpoint=True,\n            max_contributors=max(r["temporal_ensemble"]["contributor_count"] for r in rows),\n            identity_endpoint_count=sum(r["temporal_ensemble"]["identity_fast_path"] for r in rows),\n        ),',
    ),
]


def verify_query(config, case, index, chunk, evidence, checkpoint, *, sampling_seed):
    """Independently bind saved pixels/request/response to each generated chunk."""
    assert type(sampling_seed) is int and sampling_seed in config["sampling_seeds"]
    ck = next(c for c in config["checkpoints"] if c["name"] == checkpoint)
    assert ck["gpu"] in (4, 5, 6, 7)
    expected_checkpoint = (Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model").resolve()
    request_path, response_path = (case / f"{kind}_{index:04d}.json" for kind in ("request", "response"))
    evidence.check(request_path, chunk["request_sha256"])
    evidence.check(response_path, chunk["response_sha256"])
    request, response = evidence.json(request_path), evidence.json(response_path)
    assert response["status"] == "success"
    assert request["model_mode"] == "wam" and request["seed"] == sampling_seed
    assert request["action_chunk_size"] == config["online_chunk_size"]
    assert request["fps"] == config["control_fps"] and request["num_frames"] == config["online_chunk_size"] + 1
    assert request["num_steps"] == config["sampling_steps"] and request["guidance"] == config["guidance"]
    assert request["shift"] == config["shift"]
    assert request["prompt"] == evidence.json(case / "manifest.json")["record"]["task"]
    extra = request["extra"]["action128"]
    mask = [True] * 10 + [False] * 118
    assert extra["action_dim_mask"] == mask and all(type(v) is bool for v in extra["action_dim_mask"])
    template = evidence.json(config["policy_template"])
    assert extra["action_domain"] == template["action_domain"]
    assert extra["action_fps"] == config["control_fps"] and extra["return_video"] is False
    assert extra["selected_roles"] == ["auxiliary_0", "left_wrist"]
    assert extra["camera_layout"] == [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]]
    image = case / f"input_{index:04d}.png"
    assert Path(request["vision_path"]).resolve() == image.resolve()
    evidence.read(image)
    pixels = np.asarray(Image.open(image))
    assert pixels.shape == (960, 640, 3) and pixels.dtype == np.uint8
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == chunk["image_sha256"]
    args = response["args"]
    for key in ("seed", "fps", "num_frames", "action_chunk_size", "guidance", "num_steps", "shift", "prompt"):
        assert args[key] == request[key], key
    assert args["condition_frame_indexes_vision"] == [0] and args["action_path"] is None
    assert args["native_prompt_upsampling"] is False and args["model_mode"] == "wam"
    content = response["outputs"][0]["content"]
    identity = content["action128"]
    assert Path(identity["checkpoint_path"]).resolve() == expected_checkpoint
    assert type(identity["seed"]) is int and identity["seed"] == request["seed"]
    assert identity["action_domain"] == extra["action_domain"] and identity["action_dim_mask"] == mask
    assert all(type(v) is bool for v in identity["action_dim_mask"])
    assert identity["action_units"] == "raw_physical_or_source_units"
    assert identity["action_fps"] == config["control_fps"]
    close(identity["action_timestamps_seconds"], np.arange(config["online_chunk_size"]) / config["control_fps"])
    mapping(chunk["response_identity"], {k: v for k, v in identity.items() if k != "normalized_action"})
    np.testing.assert_array_equal(content["action"], chunk["raw128"])
    assert len(content["action"]) == config["online_chunk_size"]
    saved_config = evidence.yaml(Path(ck["run"]) / "config.yaml")
    spec = saved_config["model"]["config"]["action128_spec"]
    transform = spec["transforms"][spec["registry"]["domains"].index(extra["action_domain"])]
    normalized = np.asarray(identity["normalized_action"], dtype=np.float32)
    assert normalized.shape == (config["online_chunk_size"], 128) and np.isfinite(normalized).all()
    assert not normalized[:, 10:].any()
    expected = normalized.copy()
    normalization_mask = np.asarray(transform["normalization_mask"], dtype=bool)
    expected[:, normalization_mask] = (
        expected[:, normalization_mask] - np.asarray(transform["bias"], dtype=np.float32)[normalization_mask]
    ) / np.asarray(transform["scale"], dtype=np.float32)[normalization_mask]
    expected[:, 10:] = 0
    close(content["action"], expected, tolerance=1e-8)


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
    assert manifest["schema_version"] == "s2a.temporal-equal-execution.v1"
    assert manifest["reference_ensemble"] == config["reference_ensemble"]
    assert manifest["experiment"] == config and manifest["variant"] == variant and manifest["record"] == record
    assert result["sampling_seed"] == sampling_seed and result["variant"] == variant["name"]
    assert result["checkpoint"] == checkpoint and result["model_rotation_alpha"] == variant["model_rotation_alpha"]
    raw_flag = False
    assert manifest["raw_checkpoint_reference"] is raw_flag and result["raw_checkpoint_reference"] is raw_flag
    assert manifest["reference_intervention"] == result["reference_intervention"] == "causal_temporal_equal_ensemble"
    assert manifest["oracle_is_wam_success_evidence"] is False and result["oracle_is_wam_success_evidence"] is False
    assert manifest["inference_latency_excluded"] is True and result["inference_latency_excluded"] is True
    assert manifest["environment"]["CUDA_VISIBLE_DEVICES"] == manifest["environment"]["LD_LIBRARY_PATH"] == ""
    for path, signature in manifest["execution_sha256"].items():
        evidence.check(path, signature)
    required_runtime = [
        "src/s2a/temporal_ensemble.py",
        "src/s2a/temporal_ensemble_simulation.py",
        "scripts/run_wam_temporal_matrix.py",
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
            ensemble_pose, ensemble_opening, ensemble_provenance = reconstruct_ensemble(step, chunks)
            close(desired_tcp, ensemble_pose)
            close(row["opening_target"], ensemble_opening)
            compare_ensemble_log(row["temporal_ensemble"], ensemble_provenance)
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
        first_query003_regression=first_query_validator(
            config, record, variant, case, chunks[0], evidence, checkpoint, sampling_seed=sampling_seed
        )
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
        raw_checkpoint_evidence=False,
        reference_intervention="causal_temporal_equal_ensemble",
        independent_temporal_ensemble_replay=dict(
            passed=True,
            verified_endpoints=600,
            same_absolute_endpoint=True,
            max_contributors=max(r["temporal_ensemble"]["contributor_count"] for r in rows),
            identity_endpoint_count=sum(r["temporal_ensemble"]["identity_fast_path"] for r in rows),
        ),
    )
