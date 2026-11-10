"""Bind earlier task-6 feedback to completed history and its current chunk."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import prepare_task2_warning_feedback as prior
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/task2_earlier_preview_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/task2_earlier_preview_feedback_20261111.md"
BOUND = {
    "path": str(REPO / "outputs/first_step_position_bound_20261111/result.json"),
    "sha256": "71237bfad09a7dfb352a1a71fe6ba8ec51d9bdc371570956565abe7edbd8a48f",
}
CASES = [
    dict(
        case_index=10, case_id="task2_init0_s43_raw", prefix_steps=32, chunk_step=32, cursor=0, target="cream_cheese_1"
    ),
    dict(
        case_index=11,
        case_id="task2_init0_s43_terminal058",
        prefix_steps=80,
        chunk_step=80,
        cursor=0,
        target="cream_cheese_1",
    ),
]
BRANCHES = tuple(
    (case["case_index"], branch) for case in CASES for branch in ("original", "first_quality", "full_quality")
)


def source_refs():
    return prior.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "tests/research/native_runtime/test_task2_earlier_preview_inputs.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == dict(
            schema="s2a.task2-earlier-preview-feedback.v1",
            cases=CASES,
            branches=["original", "first_quality", "full_quality"],
            task_id=6,
            seed=43,
            continuation_steps=8,
            control_dt_s=0.05,
            inner_dt_s=0.002,
            joint_buffer_rad=0.05,
            position_cap_m=0.01,
            rotation_cap_rad=0.05,
            arm_rate_limit=0.25,
            braking_time_s=0.1,
            first_quality_response_budget=160,
            full_quality_response_budget=320,
            max_control_steps_per_phase=384,
            max_native_steps_per_phase=9600,
            max_models_per_phase=18,
            max_contexts_per_phase=18,
            max_refiner_calls_per_phase=48,
            max_refiner_response_evaluations_per_phase=23040,
            max_quality_calls_per_phase=32,
            max_quality_response_evaluations_per_phase=7680,
            max_audit_endpoint_queries_per_phase=6400,
            max_audit_transition_steps_per_phase=83200,
            max_workers=6,
            worker_timeout_seconds=1800,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
        "Exact earlier task2 references, policies and physical budgets",
    )
    return ref, config


def completed_prefix(case, spec):
    """Expose only endpoints through the current step and completed commands."""
    n = spec["prefix_steps"]
    old = case["intervention_spec"]
    prefix.require(
        old["case_index"] == spec["case_index"]
        and old["case_id"] == spec["case_id"]
        and old["prefix_steps"] == n + 2
        and old["chunk_step"] == spec["chunk_step"] == n
        and old["cursor"] == 2
        and spec["cursor"] == 0,
        "Earlier boundary is two steps before the verified warning within the same generated chunk",
    )
    saved = {}
    for name, value in case["saved"].items():
        prefix.require(len(value) >= n + 1, "Saved endpoint prefix is complete")
        saved[name] = (
            [np.asarray(item).copy() for item in value[: n + 1]]
            if isinstance(value, list)
            else np.asarray(value[: n + 1]).copy()
        )
    commands = np.asarray(case["commands"][:n]).copy()
    rows, receipts = case["rows"][:n], case["receipts"][: n + 1]
    prefix.require(
        commands.shape == (n, 7)
        and len(rows) == n
        and len(receipts) == n + 1
        and rows[-1]["step"] == n - 1
        and np.array_equal(rows[-1]["command"], commands[-1])
        and np.asarray(case["targets"]).shape == (40, 4, 4),
        "Only completed actions and the current generated chunk are bound",
    )
    original_spec = dict(case["spec"], prefix_steps=n)
    return dict(
        spec=original_spec,
        config=case["config"],
        record=case["record"],
        record_xml=case["record_xml"],
        commands=commands,
        saved=saved,
        receipts=receipts,
        controller=case["controller"],
        intervention_spec=spec,
        rows=rows,
        tool=case["tool"],
        targets=case["targets"],
        anchor=case["anchor"],
        fixed_targets=np.asarray(case["targets"][:8]).copy(),
    )


def load_inputs():
    config_ref, config = read_config()
    result = prefix.read(BOUND)
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["control_qualified"] is False,
        "Closed prior position-bound study",
    )
    raw_result = prefix.read(result["case_refs"][2])
    prefix.require(
        raw_result["case"]["case_index"] == 10 and raw_result["data"]["exceeds_original_position_cap"] is True,
        "Task2 raw warning has the registered model bottleneck",
    )
    prior_inputs = prior.load_inputs()
    refs = dict(prior_inputs["source_sha256"])
    refs[config_ref["path"]], refs[BOUND["path"]] = config_ref["sha256"], BOUND["sha256"]
    prefix.read(result["case_refs"][2], refs)
    cases = [completed_prefix(case, spec) for case, spec in zip(prior_inputs["cases"], CASES, strict=True)]
    return dict(
        config_ref=config_ref,
        feedback_config=config,
        cases=cases,
        asset_refs=prior_inputs["asset_refs"],
        cv2_admission=prior_inputs["cv2_admission"],
        source_sha256=refs,
    )


def policy_case(case):
    """Remove scoring identity, runtime record, and historical rows from online input."""
    prepared = prior.producer_case(case)
    keys = ("tool", "targets", "anchor", "completed_history", "previous_evidence")
    controller = {key: prepared["controller"][key] for key in ("source_from_site", "contract")}
    return {
        **{key: prepared[key] for key in keys},
        "spec": {"prefix_steps": case["spec"]["prefix_steps"]},
        "controller": controller,
        "intervention_spec": {"cursor": case["intervention_spec"]["cursor"]},
    }


def input_receipt(inputs):
    return dict(
        config_ref=inputs["config_ref"],
        bound_ref=BOUND,
        cases=[case["intervention_spec"] for case in inputs["cases"]],
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )
