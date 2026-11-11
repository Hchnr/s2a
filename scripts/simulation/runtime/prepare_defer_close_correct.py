"""Bind one correct-reference release point to its original replay evidence."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.research.native_runtime.development_prefix_replay import map_record_xml, saved_prefix
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import bind_bounded_release_shadow as cohort_binding

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/gripper/defer_close_correct_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/gripper/defer_close_correct_feedback_20261111.md"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            REPO / "tests/research/native_runtime/test_defer_close_correct_inputs.py",
            REPO / "src/s2a/research/native_runtime/task2_seed42_correct_prefix.py",
            Path(cohort_binding.__file__),
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config["schema"] == "s2a.defer-close-correct-feedback.v1"
        and config["output"] == "outputs/defer_close_correct_feedback_20261111"
        and config["case_index"] == 8
        and config["task_id"] == 6
        and config["seed"] == 42
        and config["prefix_steps"] == 80
        and config["continuation_steps"] == 8
        and config["intervention_offset"] == 5
        and config["branches"] == ["baseline", "deferred"]
        and config["target_object"] == "cream_cheese_1"
        and config["control_dt_s"] == 0.05
        and config["inner_dt_s"] == 0.002
        and config["joint_buffer_rad"] == 0.05
        and config["position_cap_m"] == 0.01
        and config["rotation_cap_rad"] == 0.05
        and config["arm_rate_limit"] == 0.25
        and config["max_control_steps_per_phase"] == 176
        and config["max_native_steps_per_phase"] == 4400
        and config["max_workers"] == 2
        and config["world_model_calls"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "fixed correct-reference feedback protocol required",
    )
    return ref, config


def configured_ref(config, name):
    ref = {"path": str(REPO / config[name]), "sha256": config[name + "_sha256"]}
    prefix.checked(ref)
    return ref


def load_inputs():
    config_ref, config = read_config()
    refs = {
        name: configured_ref(config, name)
        for name in ("prepared", "shadow", "baseline", "rows", "chunks", "arrays", "endpoints", "baseline_checker")
    }
    prepared, shadow, baseline, checked = (
        prefix.read(refs[name]) for name in ("prepared", "shadow", "baseline", "baseline_checker")
    )
    prefix.require(
        prepared["complete"] is prepared["passed"] is True
        and shadow["complete"] is shadow["passed"] is True
        and shadow["correct_steps"] == 4907
        and shadow["correct_deferred"] == 1
        and baseline["complete"] is baseline["passed"] is baseline["performance_qualified"] is True
        and baseline["case_index"] == 8
        and baseline["physics_steps"] == 182
        and checked["complete"] is checked["passed"] is checked["performance_qualified"] is True
        and checked["result_ref"] == refs["baseline"]
        and baseline["arrays_ref"] == refs["arrays"],
        "closed original correct-reference and shadow evidence required",
    )
    cohort = cohort_binding.load_inputs()
    item = cohort["cases"][8]
    prefix.require(
        item["case_id"] == "correct_08"
        and item["family"] == "correct_reference"
        and item["steps"] == 182
        and item["refs"]["result"] == refs["baseline"]
        and item["refs"]["rows"] == refs["rows"]
        and item["refs"]["chunks"] == refs["chunks"]
        and item["refs"]["native_check"] == refs["baseline_checker"],
        "the original qualified case and all baseline records must match",
    )
    shadow_case = prefix.read(shadow["case_refs"][8])
    prefix.require(
        shadow_case["case_id"] == "correct_08"
        and shadow_case["input_refs"] == item["refs"]
        and shadow_case["deferred"] == 1
        and len(shadow_case["triggers"]) == 1
        and shadow_case["triggers"][0]["step"] == 85
        and shadow_case["triggers"][0]["baseline_command"] == 1.0
        and shadow_case["triggers"][0]["proposed_command"] == 0.0,
        "the unique original correct-reference shadow trigger required",
    )
    entry = prepared["profiles"][8]
    prefix.require(entry["case_index"] == 8 and entry["profile"]["physics_steps"] == 182, "original case profile")
    manifest = prefix.read(entry["manifest_ref"])
    controller = prefix.read(entry["controller_ref"])
    tool = np.asarray(prefix.read(entry["tool_ref"])["source_from_tcp"])
    runtime_config = dict(manifest["runtime_config"])
    prefix.require(
        runtime_config["task_id"] == 6
        and runtime_config["seed"] == 42
        and runtime_config["camera_observations"] is False
        and runtime_config["refresh_controller_after_restore"] is True
        and runtime_config["render_resolution"] == 256
        and entry["tool_ref"] == item["refs"]["tool"],
        "original task, seed, controller refresh and TCP contract",
    )
    rows, chunks = prefix.read(refs["rows"]), prefix.read(refs["chunks"])
    prefix.require(len(rows) == 182 and len(chunks) == 23 and chunks[10]["step"] == 80, "complete reference timeline")
    targets, openings = np.asarray(chunks[10]["effective_targets_tcp"]), np.asarray(chunks[10]["openings"])
    prefix.require(
        targets.shape == (40, 4, 4)
        and openings.shape == (40,)
        and np.isfinite(targets).all()
        and np.all((openings >= 0) & (openings <= 1)),
        "one already-generated correct-reference chunk required",
    )
    commands = np.asarray([row["command"] for row in rows[:88]], dtype=float)
    prefix.require(commands.shape == (88, 7) and np.isfinite(commands).all(), "fixed original diagnostic commands")
    np.testing.assert_allclose(commands[85, 6], 1.0, atol=0, rtol=0)
    np.testing.assert_allclose(rows[85]["gripper"]["command"], 1.0, atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray(rows[80]["measured_source"]) @ tool, chunks[10]["anchor_tcp"], atol=1e-10, rtol=0
    )
    with np.load(prefix.checked(refs["arrays"]), allow_pickle=False) as archive:
        saved = saved_prefix(archive, 80)
        continuation_states = np.asarray(archive["sim_states"][81:89]).copy()
    prefix.require(
        hashlib.sha256(np.asarray(saved["sim_states"][0], dtype=np.float64).tobytes()).hexdigest()
        == manifest["record"]["initial_state_sha256"]
        and continuation_states.shape == (8, 79),
        "original initial state and eight scored endpoints",
    )
    receipts = prefix.read_recorded_prefix(prefix.checked(refs["endpoints"]), 81, 183)
    template_data = prefix.load_inputs({2: 16, 6: 16})
    template = template_data["cases"][0]
    record_xml = map_record_xml(
        manifest["record"]["model_xml"].encode(),
        template["record"]["model_xml"].encode(),
        template["record_xml"].encode(),
    ).decode()
    migrated = prefix.read(prefix.MIGRATED_MANIFEST)
    for key in ("libero_root", "libero_config", "osmesa_library"):
        runtime_config[key] = migrated["runtime_config"][key]
        prefix.require(Path(runtime_config[key]).is_relative_to(REPO), "repository-local migrated simulator runtime")
    consumed = dict(template_data["source_sha256"])
    for ref in [
        config_ref,
        *refs.values(),
        *item["refs"].values(),
        entry["manifest_ref"],
        entry["controller_ref"],
        entry["tool_ref"],
        shadow["case_refs"][8],
        prefix.MIGRATED_MANIFEST,
    ]:
        prefix.checked(ref, consumed)
    return dict(
        config_ref=config_ref,
        config=config,
        refs=refs,
        item=item,
        spec=dict(case_index=8, case_id="correct_08", task_id=6, seed=42, prefix_steps=80),
        runtime_config=runtime_config,
        record=manifest["record"],
        record_xml=record_xml,
        commands=commands[:80].copy(),
        fixed_continuation_commands=commands[80:88].copy(),
        saved=saved,
        continuation_states=continuation_states,
        receipts=receipts,
        controller=controller,
        tool=tool,
        targets=targets,
        openings=openings,
        baseline_intervention_decision=rows[85]["gripper"],
        baseline_rows=rows[80:88],
        asset_refs=template_data["asset_refs"],
        cv2_admission=template_data["cv2_admission"],
        source_sha256=consumed,
    )


def input_receipt(inputs):
    return {
        key: inputs[key]
        for key in ("config_ref", "refs", "item", "spec", "asset_refs", "cv2_admission", "source_sha256")
    }
