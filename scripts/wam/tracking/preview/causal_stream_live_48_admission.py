"""Admit the registered six-window continuation from audited live evidence."""

import numpy as np

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import causal_stream_live_admission as live
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/causal_stream_live_48_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/causal_stream_live_48_20261111.md"
OUTPUT = ROOT / "outputs/causal_stream_live_48_20261111"
QA = ROOT / "outputs/qa/causal_stream_live_48_20261111/result.json"
DRY = ROOT / "outputs/causal_stream_live_48_dry_20261111/result.json"
GPU = live.GPU

FIXED = {
    "schema": "s2a.causal-stream-live-48.v1",
    "output": "outputs/causal_stream_live_48_20261111",
    "prior_live_audit": "outputs/causal_stream_live_audit_20261111/result.json",
    "prior_live_audit_sha256": "601e8163443677af39196c2ad05332053778800ba9f1768b7b2fe84ad2a9636d",
    "prior_live_producer": "outputs/causal_stream_live_wam_20261111/producer/result.json",
    "prior_live_producer_sha256": "200b64e82598fe3e21ae74c8b1cadbd3a2ab818eccb135afdf0c6ad7c7b4b932",
    "prior_live_checker": "outputs/causal_stream_live_wam_20261111/checker/result.json",
    "prior_live_checker_sha256": "824543f2a49ea3a68368bd0f866ac504c238841e8f5506d1a0f377ef8f684833",
    "gpu_uuid": GPU,
    "case_index": 8,
    "seed": 42,
    "task": "put the cream cheese in the bowl",
    "prefix_steps": 80,
    "window_steps": 8,
    "max_windows": 6,
    "max_model_initializations": 1,
    "max_model_requests": 6,
    "denoise_calls_per_request": 35,
    "max_denoise_calls": 210,
    "max_control_steps_per_phase": 128,
    "max_native_steps_per_phase": 3200,
    "fresh_cases": 0,
    "fits": 0,
    "worker_timeout_seconds": 2400,
    "automatic_retry": False,
}


def validate_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The six-window model and physical budgets must exactly match registration",
    )


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    inputs = live.load_inputs()
    previous_receipt = live.input_receipt(inputs)
    consumed = dict(inputs["source_sha256"])
    refs = {
        name: {"path": str(ROOT / config[name]), "sha256": config[name + "_sha256"]}
        for name in ("prior_live_audit", "prior_live_producer", "prior_live_checker")
    }
    audit, producer, checker = (prefix.read(refs[name], consumed) for name in refs)
    prefix.require(
        audit["complete"] is audit["passed"] is audit["checker_policy_isolated"] is True
        and audit["control_qualified"] is audit["qualified_stable_success"] is False
        and audit["gpu_uuid"] == GPU
        and audit["model_initializations"] == 1
        and audit["model_requests"] == 2
        and audit["denoise_calls"] == 70
        and audit["recorded_producer_native_steps"] == audit["recorded_checker_native_steps"] == 2400
        and audit["terminal_consecutive_success_periods"] == 9,
        "The exact previous live audit is required",
    )
    for summary in (producer, checker):
        prefix.require(
            summary["complete"] is summary["passed"] is summary["sources_unchanged"] is True
            and summary["gpu_released"] is True
            and summary["counts"]["control_steps"] == 96
            and summary["counts"]["native_steps"] == 2400,
            "Both previous physical phases must be closed",
        )
    branch = prefix.read(producer["branch_ref"], consumed)
    checked = prefix.read(checker["branch_ref"], consumed)
    prefix.require(
        branch["complete"] is branch["passed"] is checked["complete"] is checked["passed"] is True
        and checked["candidate_modules_unloaded"] is True
        and len(audit["rows"]) == len(inputs["records"]) == 2,
        "The prior live branches and both response rows are required",
    )
    captures = branch["runtime"]["streaming_captures"]
    for index, (row, record, capture) in enumerate(zip(audit["rows"], inputs["records"], captures, strict=True)):
        response = prefix.read(row["actual_response_ref"], consumed)
        arrays = repro.response_arrays(response)
        prefix.require(
            row["step"] == record["step"] == capture["frame_index"] == 80 + 8 * index
            and row["image_sha256"] == record["image_sha256"] == capture["canvas_png"]["sha256"]
            and all(np.array_equal(value, np.asarray(record[key])) for key, value in arrays.items()),
            "The previous live observations and actual response arrays must match both fixed gates",
        )
    prefix.checked(config_ref, consumed)
    prefix.reference(PLAN)
    return {
        **inputs,
        "previous_live_receipt": previous_receipt,
        "long_config_ref": config_ref,
        "long_refs": refs,
        "live_config": config,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "long_config_ref": inputs["long_config_ref"],
        "long_refs": inputs["long_refs"],
        "previous_live_receipt": inputs["previous_live_receipt"],
        "model_sha256": inputs["model_sha256"],
        "source_sha256": inputs["source_sha256"],
    }
