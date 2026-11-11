"""Bind one normal-controller correct-reference stream to audited prefix evidence."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import prepare_defer_close_correct as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/ordered_correct_stream_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/ordered_correct_stream_feedback_20261111.md"
GRIP_CONFIG = REPO / "configs/tracking/gripper/ordered_correct_history_rebuild_20261111.json"


def source_refs():
    return previous.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            GRIP_CONFIG,
            Path(__file__),
            REPO / "src/s2a/research/gripper/ordered_correct_history.py",
            REPO / "src/s2a/research/joint_control/preview/ordered_correct_arm_history.py",
            REPO / "src/s2a/research/joint_control/preview/streaming_feedback_boundary.py",
            REPO / "src/s2a/research/response_models/causal/segmented_inner_error.py",
            REPO / "src/s2a/research/gripper/defer_close_transaction.py",
            REPO / "src/s2a/research/gripper/defer_close_release.py",
            REPO / "scripts/verification/reference_ports/check_defer_close_release.py",
            REPO / "tests/research/native_runtime/test_ordered_correct_stream_inputs.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    fixed = {
        "schema": "s2a.ordered-correct-stream-feedback.v1",
        "output": "outputs/ordered_correct_stream_feedback_20261111",
        "case_index": 8,
        "task_id": 6,
        "seed": 42,
        "prefix_steps": 80,
        "continuation_steps": 16,
        "window_steps": 8,
        "target_object": "cream_cheese_1",
        "control_dt_s": 0.05,
        "inner_dt_s": 0.002,
        "joint_buffer_rad": 0.05,
        "position_cap_m": 0.01,
        "rotation_cap_rad": 0.05,
        "arm_rate_limit": 0.25,
        "max_control_steps_per_phase": 96,
        "max_native_steps_per_phase": 2400,
        "max_baseline_responses_per_period": 480,
        "max_quality_responses_per_period": 160,
        "max_gripper_sequences_per_period": 6561,
        "worker_timeout_seconds": 1800,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    prefix.require(
        all(config.get(k) == v and type(config.get(k)) is type(v) for k, v in fixed.items()),
        "Fixed correct-reference feedback budget required",
    )
    return ref, config


def configured_ref(config, name):
    ref = {"path": str(REPO / config[name]), "sha256": config[name + "_sha256"]}
    prefix.checked(ref)
    return ref


def load_inputs():
    config_ref, config = read_config()
    previous_inputs = previous.load_inputs()
    grip_config = prefix.read(prefix.reference(GRIP_CONFIG))
    refs = {key: configured_ref(config, key) for key in ("gripper_rebuild", "arm_rebuild", "normal_shadow")}
    grip_audit, arm_audit, shadow = (prefix.read(refs[k]) for k in ("gripper_rebuild", "arm_rebuild", "normal_shadow"))
    prefix.require(
        grip_audit["complete"] is grip_audit["passed"] is arm_audit["complete"] is arm_audit["passed"] is True
        and grip_audit["history"]["steps"] == arm_audit["completed_periods"] == 80
        and grip_audit["history"]["max_recorded_difference"] == 0.0
        and max(arm_audit["max_innovation_difference"].values()) == 0.0
        and shadow["complete"] is shadow["passed"] is True
        and shadow["correct_cases"] == 18
        and shadow["correct_steps"] == 4907,
        "Prior completed normal-controller history and reference audits required",
    )
    normal_refs = {key: configured_ref(grip_config, key) for key in ("prepared", "rows", "chunks")}
    prepared, rows, chunks = (prefix.read(normal_refs[k]) for k in ("prepared", "rows", "chunks"))
    entry = prepared["profiles"][8]
    control, execution, tool = (prefix.read(entry[k]) for k in ("controller_ref", "execution_config_ref", "tool_ref"))
    parameters = prefix.read(execution["response_parameters_ref"])
    prefix.require(
        len(rows) == 182
        and len(chunks) == 23
        and rows[79]["step"] == 79
        and rows[80]["step"] == 80
        and chunks[10]["step"] == 80
        and chunks[11]["step"] == 88
        and control == previous_inputs["controller"]
        and np.array_equal(np.asarray(tool["source_from_tcp"]), previous_inputs["tool"]),
        "The original normal-controller case and fixed interface must match the physical replay",
    )
    commands = np.asarray([row["command"] for row in rows[:80]], dtype=float)
    np.testing.assert_array_equal(commands, previous_inputs["commands"])
    windows = chunks[10:12]
    targets, openings = (
        [np.asarray(window[k], dtype=float) for window in windows] for k in ("effective_targets_tcp", "openings")
    )
    prefix.require(
        all(x.shape == (40, 4, 4) and np.isfinite(x).all() for x in targets)
        and all(x.shape == (40,) and np.isfinite(x).all() and np.all((x >= 0) & (x <= 1)) for x in openings),
        "Two complete finite correct-reference windows required",
    )
    np.testing.assert_allclose(targets[0][8:], targets[1][:32], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(openings[0][8:], openings[1][:32])
    consumed = dict(previous_inputs["source_sha256"])
    for ref in [
        config_ref,
        prefix.reference(GRIP_CONFIG),
        *refs.values(),
        *normal_refs.values(),
        entry["controller_ref"],
        entry["execution_config_ref"],
        entry["tool_ref"],
        execution["response_parameters_ref"],
    ]:
        prefix.checked(ref, consumed)
    return {
        "config_ref": config_ref,
        "config": config,
        "previous": previous_inputs,
        "refs": refs,
        "normal_refs": normal_refs,
        "rows": rows[:80],
        "tracking_before": rows[80]["refinement"]["history_before"],
        "preaction_site": rows[80]["site_pose"],
        "preaction_twist": rows[80]["site_twist"],
        "prefix_chunks": chunks[:10],
        "targets": targets,
        "openings": openings,
        "first_anchor": windows[0]["anchor_tcp"],
        "parameters": parameters,
        "velocity_envelope": execution["gripper_velocity_envelope"],
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "normal_refs": inputs["normal_refs"],
        "prior": previous.input_receipt(inputs["previous"]),
        "source_sha256": inputs["source_sha256"],
    }
