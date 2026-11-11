"""Bind a saved-response physical probe to independently checked live evidence."""

import numpy as np

from scripts.simulation.runtime import causal_stream_live_48_replay as references
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.verification.reference_ports import audit_causal_stream_live_cached as live_audit
from scripts.wam.tracking.preview import causal_stream_live_cached_admission as live

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/quality_rate_repair_recorded_feedback_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/quality_rate_repair_recorded_feedback_20261111.md"
OUTPUT = ROOT / "outputs/quality_rate_repair_recorded_feedback_20261111"
QA = ROOT / "outputs/qa/quality_rate_repair_recorded_feedback_20261111/result.json"
DRY = ROOT / "outputs/quality_rate_repair_recorded_feedback_dry_20261111/result.json"
FIXED = {
    "schema": "s2a.quality-rate-repair-recorded-feedback.v1",
    "output": "outputs/quality_rate_repair_recorded_feedback_20261111",
    "live_audit": "outputs/causal_stream_live_cached_audit_20261111/result.json",
    "live_audit_sha256": "65f51ff8b9135ba01462ec96f662ed648024eb1df5e8dee1b3fef09dbaaa87c9",
    "solver_static": "outputs/quality_rate_repair_solver_20261111/worker/result.json",
    "solver_static_sha256": "fb9824cba9ce18fc4acb52bcb25c336d8fa173522c3d27be66bc4acbfe7d9a26",
    "case_index": 8,
    "seed": 42,
    "prefix_steps": 80,
    "window_steps": 8,
    "max_windows": 6,
    "expected_prior_periods": 45,
    "max_control_steps_per_phase": 128,
    "max_native_steps_per_phase": 3200,
    "max_baseline_model_responses": 23040,
    "max_quality_model_responses": 7680,
    "max_independent_endpoint_queries": 9600,
    "max_independent_transition_steps": 124800,
    "max_gripper_candidates": 314928,
    "new_model_initializations": 0,
    "new_model_requests": 0,
    "new_denoise_calls": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 2400,
    "automatic_retry": False,
}


def validate_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The exact saved-reference physical budget is required",
    )


def periods(branch):
    return [
        row for segment in branch["runtime"]["continuation"]["segments"] for row in segment["continuation"]["periods"]
    ]


def matched_prior_period(actual, expected):
    """Compare only completed commands and measured physics, never task labels."""
    prefix.require(actual["step"] == expected["step"], "Completed physical period identity changed")
    for key in ("command", "actual_state"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=0)
    prefix.require(len(actual["actual_inner"]) == len(expected["actual_inner"]) == 25, "Twenty-five inner samples")
    for key in ("q", "qvel", "time"):
        np.testing.assert_allclose(
            [item[key] for item in actual["actual_inner"]],
            [item[key] for item in expected["actual_inner"]],
            atol=1e-10,
            rtol=0,
        )
    return True


def build_records(inputs, branch, consumed):
    """Verify each historical image/request/response before constructing a relay roster."""
    image_root = live_audit.RUN / "producer/simulation/runtime"
    chunks, sources = references.decode_actual_references(inputs, branch, image_root=image_root, max_windows=6)
    prefix.require(len(chunks) == len(sources) == 6, "All six source-verified WAM references are required")
    runtime = branch["runtime"]
    records = []
    for index, (capture, exchange) in enumerate(
        zip(runtime["streaming_captures"], runtime["streaming_model_exchanges"], strict=True)
    ):
        request = prefix.read(exchange["request_ref"], consumed)
        response = prefix.read(exchange["response_ref"], consumed)
        prefix.checked(capture["canvas_png"], consumed)
        prefix.require(
            request["request_id"] == index
            and request["step"] == response["step"] == 80 + 8 * index
            and request["image"]["sha256"] == response["image_sha256"] == capture["canvas_png"]["sha256"]
            and response["denoise_calls"] == 35
            and response["action_dim_mask"] == [True] * 10 + [False] * 118,
            "A real historical response must bind the exact current image and original mask",
        )
        records.append(
            {
                "step": request["step"],
                "sim_time_s": request["sim_time_s"],
                "state_sha256": request["state_sha256"],
                "image_sha256": request["image"]["sha256"],
                "response_sha256": exchange["response_ref"]["sha256"],
                "historical_denoise_calls": response["denoise_calls"],
                "action": response["action"],
                "normalized_action": response["normalized_action"],
            }
        )
    return records


def load_inputs():
    inputs = live.load_inputs()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    consumed = dict(inputs["source_sha256"])
    refs = {
        name: {"path": str(ROOT / config[name]), "sha256": config[name + "_sha256"]}
        for name in ("live_audit", "solver_static")
    }
    saved_audit, static = (prefix.read(refs[name], consumed) for name in ("live_audit", "solver_static"))
    prefix.require(
        saved_audit["complete"] is saved_audit["passed"] is saved_audit["checker_policy_isolated"] is True
        and saved_audit["model_requests"] == 6
        and saved_audit["denoise_calls"] == 210
        and saved_audit["producer_stop"]["at_step"] == 125
        and static["complete"] is static["passed"] is True
        and static["analysis"]["candidates"]["quality_rate_repair"]["certified_model_witness"] is True,
        "Prior source, independent replay and static witness must be complete",
    )
    branches, branch_refs = {}, {}
    for phase, digest in (("producer", live_audit.PRODUCER_SHA), ("checker", live_audit.CHECKER_SHA)):
        summary_ref = {"path": str(live_audit.RUN / phase / "result.json"), "sha256": digest}
        summary = prefix.read(summary_ref, consumed)
        prefix.require(summary["complete"] is summary["passed"] is True, "Closed prior live phase required")
        branch_refs[phase] = summary["branch_ref"]
        branches[phase] = prefix.read(summary["branch_ref"], consumed)
    original, replay = branches["producer"], branches["checker"]
    produced, replayed = periods(original), periods(replay)
    prefix.require(
        len(produced) == len(replayed) == 45
        and [row["step"] for row in produced] == list(range(80, 125))
        and original["runtime"]["continuation"]["stop"]["reason"] == "quality_no_witness"
        and replay["candidate_modules_unloaded"] is True,
        "Forty-five previously replayed periods and the original quality stop are required",
    )
    for actual, checked in zip(produced, replayed, strict=True):
        historical.compare_first_period(actual, checked)
    records = build_records(inputs, original, consumed)
    prefix.checked(config_ref, consumed)
    prefix.reference(PLAN)
    inputs.update(
        feedback_config=config,
        feedback_config_ref=config_ref,
        feedback_refs=refs,
        source_sha256=consumed,
        records=records,
        prior_periods=produced,
        prior_live_branch_refs=branch_refs,
    )
    return inputs


def input_receipt(inputs):
    return {
        "feedback_config_ref": inputs["feedback_config_ref"],
        "feedback_refs": inputs["feedback_refs"],
        "prior_live_branch_refs": inputs["prior_live_branch_refs"],
        "record_response_sha256": [record["response_sha256"] for record in inputs["records"]],
        "model_sha256": inputs["model_sha256"],
        "source_sha256": inputs["source_sha256"],
    }
