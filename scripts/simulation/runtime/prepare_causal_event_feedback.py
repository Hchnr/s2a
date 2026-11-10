"""Bind matched generated references to the two reproduced causal prefixes."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import compare_multicase_emergency_retreat as saved_chunk
from scripts.simulation.runtime import replay_causal_event_prefixes as replay
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_event_feedback_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_event_feedback_20261110.md"
GRIPPER_RESULT = {
    "path": str(REPO / "outputs/causal_gripper_history_20261110/result.json"),
    "sha256": "7e919a511ab13dedba478d521cb52e13a2294b6364e46b0b76f049567abf168b",
}
PREFIX_RESULT = {
    "path": str(REPO / "outputs/causal_event_prefix_replay_20261110/result.json"),
    "sha256": "86a33474a493ce7c5d65cd465cd73efcb731a01411c52cc3644b69f8bc053f65",
}
CASES = [
    {
        "case_index": 0,
        "case_id": "task0_init0_s42_raw",
        "prefix_steps": 136,
        "old_step": 128,
        "task_id": 8,
        "target": "akita_black_bowl_1",
    },
    {
        "case_index": 5,
        "case_id": "task1_init0_s42_terminal058",
        "prefix_steps": 360,
        "old_step": 344,
        "task_id": 9,
        "target": "wine_bottle_1",
    },
]


def source_refs():
    return replay.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            Path(saved_chunk.__file__),
            REPO / "tests/research/native_runtime/test_causal_event_feedback_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.causal-event-feedback.v1",
            "cases": CASES,
            "branches": ["latest", "remembered"],
            "seed": 42,
            "continuation_steps": 8,
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "comparison_atol": 1e-10,
            "comparison_rtol": 0,
            "max_control_steps_per_phase": 1024,
            "max_native_steps_per_phase": 25600,
            "max_models_per_phase": 12,
            "max_contexts_per_phase": 12,
            "max_refiner_calls_per_phase": 32,
            "max_refiner_response_evaluations_per_phase": 15360,
            "max_quality_calls_per_phase": 32,
            "max_quality_response_evaluations_per_phase": 5120,
            "max_gripper_planner_calls_per_phase": 1024,
            "max_gripper_candidate_sequences_per_phase": 6718464,
            "max_workers": 4,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered matched feedback scope and budget",
    )
    return ref, config


def validate_references(spec, old, latest, shadow, measured_tcp):
    n, old_step = spec["prefix_steps"], spec["old_step"]
    prefix.require(
        old["step"] == old_step and latest["step"] == n and old_step < n, "Both already generated references"
    )
    references = {}
    for name, chunk, cursor in (("remembered", old, n - old_step), ("latest", latest, 0)):
        poses = np.asarray(chunk["effective_targets_tcp"], dtype=float)
        openings = np.asarray(chunk["openings"], dtype=float)
        anchor = np.asarray(chunk["anchor_tcp"], dtype=float)
        prefix.require(
            poses.shape == (40, 4, 4)
            and openings.shape == (40,)
            and anchor.shape == (4, 4)
            and np.isfinite(poses).all()
            and np.isfinite(openings).all()
            and np.isfinite(anchor).all()
            and np.all((openings >= 0) & (openings <= 1))
            and 0 <= cursor <= 32,
            "Complete finite paired pose/opening reference and valid cursor",
        )
        references[name] = {"poses": poses, "openings": openings, "anchor": anchor, "cursor": cursor}
    np.testing.assert_allclose(references["latest"]["anchor"], measured_tcp, atol=1e-10, rtol=0)
    memory = references["remembered"]
    indices = list(range(memory["cursor"], memory["cursor"] + 8))
    prefix.require(
        shadow["source"] == "remembered"
        and shadow["step"] == n
        and shadow["source_step"] == old_step
        and shadow["source_indices"] == indices
        and shadow["absolute_endpoints"] == list(range(n + 1, n + 9))
        and shadow["event"]["direction"] == "closure",
        "The independently checked first closure uses the same absolute endpoints",
    )
    np.testing.assert_array_equal(shadow["poses"], memory["poses"][indices])
    np.testing.assert_array_equal(shadow["openings"], memory["openings"][indices])
    return references


def selected_case(case, mode):
    """Give the policy only the one chosen reference; other scores stay external."""
    prefix.require(mode in ("latest", "remembered"), "One registered matched branch")
    reference = case["references"][mode]
    cursor = reference["cursor"]
    return {
        "spec": case["spec"],
        "controller": case["controller"],
        "tool": case["tool"].copy(),
        "targets": reference["poses"].copy(),
        "openings": reference["openings"].copy(),
        "anchor": reference["anchor"].copy(),
        "cursor": cursor,
        "fixed_targets": reference["poses"][cursor : cursor + 8].copy(),
        "fixed_openings": reference["openings"][cursor : cursor + 8].copy(),
    }


def load_inputs():
    config_ref, config = read_config()
    physical, gripper = prefix.read(PREFIX_RESULT), prefix.read(GRIPPER_RESULT)
    prefix.require(
        physical["complete"]
        is physical["passed"]
        is physical["sources_unchanged"]
        is gripper["complete"]
        is gripper["passed"]
        is gripper["sources_unchanged"]
        is True
        and physical["control_steps"] == gripper["reconstructed_steps"] == 496
        and physical["newly_selected_commands"] == 0
        and len(physical["case_refs"]) == len(gripper["case_refs"]) == 2,
        "Reproduced physical and gripper histories for both registered cases",
    )
    inputs = replay.load_inputs()
    gripper_registration = prefix.read(gripper["registration_ref"])
    refs = dict(inputs["source_sha256"])
    for path, digest in gripper_registration["source_sha256"].items():
        prefix.require(path not in refs or refs[path] == digest, "Historical input identity conflict")
        refs[path] = digest
    refs.update(
        {
            config_ref["path"]: config_ref["sha256"],
            PREFIX_RESULT["path"]: PREFIX_RESULT["sha256"],
            GRIPPER_RESULT["path"]: GRIPPER_RESULT["sha256"],
            gripper["registration_ref"]["path"]: gripper["registration_ref"]["sha256"],
        }
    )
    cases = []
    for index, (spec, original, grip_spec, physical_ref, gripper_ref) in enumerate(
        zip(
            CASES,
            inputs["cases"],
            gripper_registration["cases"],
            physical["case_refs"],
            gripper["case_refs"],
            strict=True,
        )
    ):
        n = spec["prefix_steps"]
        p, g = prefix.read(physical_ref, refs), prefix.read(gripper_ref, refs)
        prefix.require(
            original["spec"]["case_index"] == grip_spec["spec"]["case_index"] == spec["case_index"]
            and original["spec"]["case_id"] == grip_spec["spec"]["case_id"] == spec["case_id"]
            and original["spec"]["prefix_steps"] == grip_spec["spec"]["prefix_steps"] == n
            and p["complete"] is p["passed"] is g["complete"] is g["passed"] is True
            and p["runtime"]["checked_endpoints"] == p["runtime"]["exact_state_endpoints"] == n + 1
            and g["data"]["steps"] == n
            and g["data"]["last_completed_command"] == float(original["commands"][-1, 6]),
            "Per-case physical state and completed gripper history agree",
        )
        for key in ("steps_ref", "chunks_ref", "tool_ref", "parameters_ref", "controller_ref"):
            prefix.checked(grip_spec[key], refs)
        tool = np.asarray(prefix.read(grip_spec["tool_ref"])["source_from_tcp"], dtype=float)
        chunk_path = prefix.checked(grip_spec["chunks_ref"])
        old = saved_chunk.read_array_item(chunk_path, spec["old_step"] // 8)
        latest = saved_chunk.read_array_item(chunk_path, n // 8)
        measured = np.asarray(original["saved"]["source_pose"][-1]) @ tool
        references = validate_references(spec, old, latest, original["opportunity"]["shadow"], measured)
        rows = prefix.read_recorded_prefix(prefix.checked(grip_spec["steps_ref"]), n, 600)
        prefix.require(
            rows[-1]["step"] == n - 1 and np.array_equal(np.asarray(rows[-1]["command"]), original["commands"][-1]),
            "Only completed previous action and history",
        )
        cases.append(
            {
                **original,
                "spec": spec,
                "tool": tool,
                "references": references,
                "rows": rows,
                "gripper_inputs": grip_spec,
                "gripper_result_ref": gripper_ref,
            }
        )
    return {
        "config_ref": config_ref,
        "config": config,
        "cases": cases,
        "asset_refs": inputs["asset_refs"],
        "source_sha256": refs,
        "cv2_admission": inputs["cv2_admission"],
    }
