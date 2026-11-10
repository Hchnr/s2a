"""Bind task-6 warning feedback to audited prefixes and available references."""

import json
from pathlib import Path

import numpy as np

from scripts.simulation.runtime import compare_multicase_emergency_retreat as chunks
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import replay_task2_warning_prefix as prior

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_warning_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_warning_feedback_20261111.md"
PREFIX = {
    "path": str(REPO / "outputs/task2_warning_prefix_20261111/result.json"),
    "sha256": "a1d84370edafbc97d072c59370c8f33a06c17600ed1baf843b1286e1449d0a27",
}
CASES = [
    dict(
        case_index=10, case_id="task2_init0_s43_raw", prefix_steps=34, chunk_step=32, cursor=2, target="cream_cheese_1"
    ),
    dict(
        case_index=11,
        case_id="task2_init0_s43_terminal058",
        prefix_steps=82,
        chunk_step=80,
        cursor=2,
        target="cream_cheese_1",
    ),
]


def source_refs():
    return prior.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "scripts/simulation/runtime/compare_multicase_emergency_retreat.py",
            REPO / "tests/research/native_runtime/test_task2_warning_feedback_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.task2-warning-feedback.v1",
            "cases": CASES,
            "branches": ["original", "quality"],
            "task_id": 6,
            "seed": 43,
            "continuation_steps": 8,
            "control_dt_s": 0.05,
            "inner_dt_s": 0.002,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "braking_time_s": 0.1,
            "absolute_comparison_tolerance": 1e-10,
            "relative_comparison_tolerance": 0,
            "max_control_steps_per_phase": 264,
            "max_native_steps_per_phase": 6600,
            "max_models_per_phase": 12,
            "max_contexts_per_phase": 12,
            "max_refiner_calls_per_phase": 32,
            "max_refiner_response_evaluations_per_phase": 15360,
            "max_quality_calls_per_phase": 16,
            "max_quality_response_evaluations_per_phase": 2560,
            "max_audit_endpoint_queries_per_phase": 3200,
            "max_audit_transition_steps_per_phase": 41600,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact task2 warning feedback cases, constraints and costs",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    result = prefix.read(PREFIX)
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["counts"]
        == dict(control_steps=116, inner_steps=2900, model_initializations=6, contexts=6, new_actions=0)
        and len(result["case_refs"]) == 2,
        "Both original task2 prefixes must be physically reproduced",
    )
    original = prior.load_inputs()
    review = prefix.read(prefix.REVIEW)
    source_map = review["source_and_artifact_sha256"]
    refs = dict(original["source_sha256"])
    refs[config_ref["path"]] = config_ref["sha256"]
    refs[PREFIX["path"]] = PREFIX["sha256"]

    def reviewed(path):
        ref = prefix.reference(path)
        prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Original reviewed reference and control history")
        return prefix.checked(ref, refs)

    cases = []
    for spec, case, case_ref in zip(CASES, original["cases"], result["case_refs"], strict=True):
        index, n = spec["case_index"], spec["prefix_steps"]
        closed = prefix.read(case_ref, refs)
        runtime = closed["runtime"]
        prefix.require(
            case["spec"]["case_index"] == index
            and case["spec"]["case_id"] == spec["case_id"]
            and case["spec"]["prefix_steps"] == n
            and closed["case"] == case["spec"]
            and closed["complete"] is closed["passed"] is runtime["behavioural_replay_verified"] is True
            and runtime["checked_endpoints"] == runtime["exact_state_endpoints"] == n + 1
            and runtime["final_prefix_inner"]["below_buffer_samples"] == 0,
            "Original prefix and all final inner samples verified",
        )
        folder = Path(review["cases"][index]["mechanical"]["case"])
        manifest = json.loads(reviewed(folder / "manifest.json").read_text())
        tool = np.asarray(
            prefix.read(prefix.reference(reviewed(Path(manifest["runtime_config"]["tool_contract"]))), refs)[
                "source_from_tcp"
            ],
            dtype=float,
        )
        rows = prefix.read_recorded_prefix(reviewed(folder / "steps.jsonl"), n, 600)
        chunk = chunks.read_array_item(reviewed(folder / "chunks.json"), spec["chunk_step"] // 8)
        poses, openings, anchor = (
            np.asarray(chunk[key], dtype=float) for key in ("effective_targets_tcp", "openings", "anchor_tcp")
        )
        cursor = spec["cursor"]
        prefix.require(
            chunk["step"] == spec["chunk_step"]
            and spec["chunk_step"] + cursor == n
            and poses.shape == (40, 4, 4)
            and openings.shape == (40,)
            and anchor.shape == (4, 4)
            and np.isfinite(poses).all()
            and np.isfinite(openings).all()
            and np.all((openings >= 0) & (openings <= 1))
            and rows[-1]["step"] == n - 1
            and np.array_equal(rows[-1]["command"], case["commands"][-1]),
            "Only completed history and currently generated reference are used",
        )
        np.testing.assert_allclose(
            anchor, np.asarray(rows[spec["chunk_step"]]["measured_source"]) @ tool, atol=1e-10, rtol=0
        )
        cases.append(
            {
                **case,
                "intervention_spec": spec,
                "rows": rows,
                "tool": tool,
                "targets": poses,
                "anchor": anchor,
                "fixed_targets": poses[cursor : cursor + 8].copy(),
                "prefix_result_ref": case_ref,
            }
        )
    return {
        **original,
        "feedback_config_ref": config_ref,
        "feedback_config": config,
        "cases": cases,
        "source_sha256": refs,
    }


def producer_case(case):
    """Construct arm history only from the actual last completed period."""
    from s2a.research.joint_control.preview.completed_history import CompletedHistory

    row, n = case["rows"][-1], case["spec"]["prefix_steps"]
    coupled = row["coupled_response_history"]
    prefix.require(row["step"] == n - 1 and coupled["completed_periods_after"] == n, "Completed causal arm history")
    history = CompletedHistory(
        previous_command=row["command"],
        coupled_innovation=coupled["innovation_after"],
        tracking_innovation=row["tracking_history_after"],
        last_plan=row["tracking_refinement"]["planned_commands"],
        completed_periods=n,
    )
    evidence = dict(
        site=row["qualified_forward_input"]["x_site_pose"][0],
        twist=row["qualified_forward_input"]["x_site_twist"][0],
        coupled_prediction=coupled["commit"]["nominal_response"],
        tracking_prediction=row["tracking_refinement"]["nominal_response"],
    )
    return {**case, "completed_history": history, "previous_evidence": evidence}


def input_receipt(inputs):
    return dict(
        config_ref=inputs["feedback_config_ref"],
        cases=[item["intervention_spec"] for item in inputs["cases"]],
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )
