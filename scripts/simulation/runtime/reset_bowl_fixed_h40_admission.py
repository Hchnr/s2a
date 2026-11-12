"""Freeze the previously audited reset and actual step8 WAM reference."""

from pathlib import Path

import numpy as np

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_multitask_live_admission_v2 as previous
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_fixed_h40_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_fixed_h40_probe_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_fixed_h40_20261112"
QA = ROOT / "outputs/qa/reset_bowl_fixed_h40_final_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_fixed_h40_dry_20261112/result.json"
REPLAY_AUDIT = {
    "path": str(ROOT / "outputs/reset_multitask_live_recovery_audit_20261111/result.json"),
    "sha256": "218582f9851373f02a3dbd22d392cb48783507dc241d4835fc6783fc8bf4e2a3",
}
INTENT_AUDIT = {
    "path": str(ROOT / "outputs/reset_multitask_gripper_intent_audit_20261112/result.json"),
    "sha256": "17d9f95a12296de25f59431fc22dce1cf1ab7ca3cc5cb7b9d6fea7bb608aa0f9",
}
RESPONSE_SHA = "c75e1a985879a44533a317943244227061c790a2a2370c5a51502c3ef612968b"
IMAGE_SHA = "1a198b0ebd94cebf964d97c7f541ff25d6a05e15fd8479e402bbbd6ca978b66a"
FIXED = dict(
    schema="s2a.reset-bowl-fixed-h40.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_index=0,
    case_id="task0_init0_s42_raw",
    task="put the bowl on the plate",
    target="akita_black_bowl_1",
    seed=42,
    reference_response_sha256=RESPONSE_SHA,
    reference_image_sha256=IMAGE_SHA,
    first_window_steps=8,
    generated_horizon=40,
    segment_steps=8,
    max_control_steps_per_phase=48,
    max_native_steps_per_phase=1200,
    max_models_per_phase=3,
    max_contexts_per_phase=3,
    max_quality_responses_per_period=320,
    max_baseline_responses_per_period=480,
    max_gripper_sequences_per_period=6561,
    max_audit_endpoint_queries=9600,
    max_audit_transition_steps=124800,
    previous_periods=276,
    previous_inner_steps=6900,
    previous_model_initializations=6,
    previous_model_requests_attempted=18,
    previous_model_requests_completed=15,
    previous_denoises=525,
    new_model_initializations=0,
    new_model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=2400,
)


def validate_config(config):
    prefix.require(
        previous.previous._exact(config, FIXED), "The fixed bowl case, saved reference and budgets are required"
    )


def source_refs():
    names = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/run_reset_bowl_fixed_h40.py",
        ROOT / "scripts/simulation/runtime/reset_bowl_fixed_h40.py",
        ROOT / "scripts/simulation/runtime/reset_bowl_fixed_h40_replay.py",
        ROOT / "src/s2a/references/finite_horizon_window.py",
        ROOT / "tests/unit/references/test_finite_horizon_window.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_fixed_h40.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_fixed_h40_replay.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_fixed_h40_admission.py",
        ROOT / "tests/research/native_runtime/test_reset_bowl_fixed_h40_owner.py",
    )
    return previous.source_refs() + [prefix.reference(path) for path in names]


def load_inputs(*, verify_assets=True):
    prefix.environment()
    config_ref, plan_ref = prefix.reference(CONFIG), prefix.reference(PLAN)
    config = prefix.read(config_ref)
    validate_config(config)
    base = previous.load_inputs(verify_assets=verify_assets)
    audit, intent = prefix.read(REPLAY_AUDIT), prefix.read(INTENT_AUDIT)
    prefix.require(
        audit["complete"] is audit["passed"] is intent["complete"] is intent["passed"] is True
        and audit["control_qualified"] is intent["control_qualified"] is False
        and intent["source_ref"] == REPLAY_AUDIT
        and audit["cumulative_periods"] == config["previous_periods"]
        and audit["cumulative_inner_steps"] == config["previous_inner_steps"]
        and audit["cumulative_model_initializations"] == config["previous_model_initializations"]
        and audit["cumulative_requests_attempted"] == config["previous_model_requests_attempted"]
        and audit["cumulative_requests_completed"] == config["previous_model_requests_completed"]
        and audit["cumulative_denoises"] == config["previous_denoises"],
        "The audited prior runs and actual cumulative costs must match",
    )
    audited = audit["cases"][0]
    prefix.require(
        audited["case_index"] == 0 and audited["physical_periods"] == 48 and audited["model_requests"] == 5,
        "The complete bowl development branch is required",
    )
    producer, model = prefix.read(audited["producer_ref"]), prefix.read(audited["model_ref"])
    case = base["selected"][0]["case"]
    prefix.require(
        producer["complete"] is producer["passed"] is model["complete"] is model["passed"] is True
        and producer["case"] == case["case"]
        and case["case"]["case_id"] == config["case_id"]
        and case["record"]["task"] == config["task"]
        and case["target_object"] == config["target"],
        "The saved response must come from the registered bowl reset",
    )
    runtime = producer["runtime"]
    capture, exchange, row = (
        runtime["streaming_captures"][0],
        runtime["streaming_model_exchanges"][0],
        model["responses"][0],
    )
    response_ref, envelope_ref = row["actual_response_ref"], exchange["response_ref"]
    response, envelope, request = (
        prefix.read(response_ref),
        prefix.read(envelope_ref),
        prefix.read(exchange["request_ref"]),
    )
    arrays = repro.response_arrays(response)
    np.testing.assert_array_equal(arrays["action"], envelope["action"])
    np.testing.assert_array_equal(arrays["normalized_action"], envelope["normalized_action"])
    prefix.require(
        response_ref["sha256"] == config["reference_response_sha256"]
        and capture["canvas_png"]["sha256"] == config["reference_image_sha256"]
        and capture["frame_index"] == exchange["step"] == request["step"] == row["step"] == 8
        and request["image"]["sha256"] == envelope["image_sha256"] == capture["canvas_png"]["sha256"]
        and request["state_sha256"] == envelope["state_sha256"] == capture["current_endpoint"]["sim_state_sha256"]
        and row["reference_sent"] is True,
        "The saved H40 must match the actual step8 image and accepted response",
    )
    prefix.checked(capture["canvas_png"])
    refs = dict(
        config=config_ref,
        plan=plan_ref,
        replay_audit=REPLAY_AUDIT,
        intent_audit=INTENT_AUDIT,
        old_producer=audited["producer_ref"],
        old_checker=audited["checker_ref"],
        old_model=audited["model_ref"],
        response=response_ref,
        envelope=envelope_ref,
        request=exchange["request_ref"],
        image=capture["canvas_png"],
    )
    consumed = dict(base["source_sha256"])
    for ref in refs.values():
        prefix.checked(ref, consumed)
    return dict(
        config=config,
        base=base,
        case=case,
        old_producer=producer,
        baseline=runtime["continuation"]["first_window"]["periods"],
        expected_capture=capture,
        refs=refs,
        source_sha256=consumed,
    )


def input_receipt(inputs):
    return dict(
        refs=inputs["refs"],
        previous=previous.input_receipt(inputs["base"]),
        case=inputs["case"]["case"],
        source_sha256=inputs["source_sha256"],
    )


def load_reference(inputs):
    return prefix.read(inputs["refs"]["response"]), prefix.read(inputs["refs"]["envelope"])
