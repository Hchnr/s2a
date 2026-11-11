"""Admit the failed live stream for one separately budgeted CPU replay."""

import copy

import numpy as np

from scripts.simulation.runtime import causal_stream_live_48_replay as references
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import causal_stream_live_48_admission as live
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_stream_requests as old_requests
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/causal_stream_endpoint_diagnostic_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/causal_stream_endpoint_diagnostic_20261111.md"
OUTPUT = ROOT / "outputs/causal_stream_endpoint_diagnostic_20261111"
QA = ROOT / "outputs/qa/causal_stream_endpoint_diagnostic_20261111/result.json"
DRY = ROOT / "outputs/causal_stream_endpoint_diagnostic_dry_20261111/result.json"
PREFLIGHT = {
    "path": str(ROOT / "outputs/causal_stream_live_48_20261111/preflight.json"),
    "sha256": "44659d32250550783f0b0929661efe539bdd8e9de7ec354cacbddaf022e30447",
}

FIXED = {
    "schema": "s2a.causal-stream-endpoint-diagnostic.v1",
    "output": "outputs/causal_stream_endpoint_diagnostic_20261111",
    "failed_owner": "outputs/causal_stream_live_48_20261111/producer/result.json",
    "failed_owner_sha256": "21f23c5160d81e0cb15aa7671fdfa2d5f092f24504b108d27c5b97c3e88bfab7",
    "failed_branch": "outputs/causal_stream_live_48_20261111/producer/simulation/result.json",
    "failed_branch_sha256": "39cdfc3c409b8f4d010548b8695182ee53c81490843234a4cf1e1ba04fb188bb",
    "failed_model": "outputs/causal_stream_live_48_20261111/producer/model/result.json",
    "failed_model_sha256": "86b3604649a4189e7a5bc34e83c1a5efc0025bdcfc12849c0d475d72d2b8bab9",
    "case_index": 8,
    "seed": 42,
    "prefix_steps": 80,
    "replayed_steps": 40,
    "max_native_steps": 3000,
    "endpoint_step": 120,
    "new_model_initializations": 0,
    "new_model_requests": 0,
    "new_denoise_calls": 0,
    "newly_selected_commands": 0,
    "fresh_cases": 0,
    "worker_timeout_seconds": 1200,
    "automatic_retry": False,
}


def validate_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The complete CPU endpoint diagnostic budget must remain fixed",
    )


def diagnostic_stream(branch):
    """Derive a replay boundary without claiming the producer sent a stop."""
    runtime = branch["runtime"]
    partial = runtime["streaming_partial"]
    prefix.require(
        branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is False
        and runtime["failure_stage"] == "continuation"
        and runtime["error"]["message"] == "Endpoint observation changed the physical state"
        and runtime["cleanup_complete"] is True
        and runtime["control_steps_attempted"] == runtime["control_steps_completed"] == 120
        and runtime["counters"]["mj_step"]["attempted"] == runtime["counters"]["mj_step"]["completed"] == 3000
        and partial["complete"] is False
        and partial["stop"] is None
        and partial["error"]["message"] == runtime["error"]["message"]
        and partial["requests_issued"] == partial["responses_accepted"] == 5
        and partial["feedback_periods_committed"] == partial["control_steps_completed"] == 40
        and len(partial["segments"]) == 5
        and all(
            row["inflight"] is None
            and row["continuation"]["steps_completed"] == len(row["continuation"]["periods"]) == 8
            and row["continuation"]["stop"] is None
            for row in partial["segments"]
        ),
        "The exact failed producer and all 40 completed actions are required",
    )
    stream = copy.deepcopy(partial)
    stream["stop"] = {
        "reason": "diagnostic_saved_action_boundary",
        "at_step": 120,
        "next_action_dispatched": False,
    }
    stream["diagnostic_derived_stop"] = True
    return stream


def verify_model(inputs, model, branch, consumed):
    prefix.require(
        model["complete"] is model["passed"] is False
        and model["error"]["type"] == "EOFError"
        and model["model_initializations"] == 1
        and model["model_requests_attempted"] == model["model_requests_completed"] == model["responses_sent"] == 5
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 175
        and len(model["responses"]) == len(model["mask_evidence"]) == 5
        and len(model["denoise_evidence"]) == 175,
        "The failed service must retain all five actual responses and 175 denoise calls",
    )
    engine.validate_mask_evidence(model["mask_evidence"], 5)
    captures = branch["runtime"]["streaming_captures"]
    exchanges = branch["runtime"]["streaming_model_exchanges"]
    for index, (row, capture, exchange) in enumerate(zip(model["responses"], captures, exchanges, strict=True)):
        step = 80 + index * 8
        response = prefix.read(row["actual_response_ref"], consumed)
        request = prefix.read(row["model_request_ref"], consumed)
        delivered = prefix.read(exchange["response_ref"], consumed)
        arrays = repro.response_arrays(response)
        old_requests._check_response_identity(
            response,
            model_request=request,
            task=inputs["live_config"]["task"],
            checkpoint=inputs["prepared"]["model_config"]["checkpoint"],
        )
        prefix.require(
            row["step"] == capture["frame_index"] == exchange["step"] == step
            and row["reference_sent"] is True
            and repro.sha256(request["vision_path"]) == capture["canvas_png"]["sha256"]
            and all(np.array_equal(value, np.asarray(delivered[key])) for key, value in arrays.items()),
            "The actual model output and current image must match the delivered reference",
        )
        calls = model["denoise_evidence"][index * 35 : (index + 1) * 35]
        prefix.require(
            all(call["completed"] is True and call["label"] == request["name"] for call in calls),
            "Every reference must have 35 completed actual denoise calls",
        )


def load_inputs():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    inputs = live.load_inputs()
    consumed = dict(inputs["source_sha256"])
    preflight = prefix.read(PREFLIGHT, consumed)
    prefix.require(
        preflight["complete"] is preflight["passed"] is True
        and preflight["gpu_uuid"] == live.GPU
        and preflight["weight_files_verified"] == 163
        and preflight["model_initializations"] == preflight["model_requests"] == preflight["physics_steps"] == 0,
        "The original zero-work source and checkpoint preflight is required",
    )
    for ref in preflight["source_refs"]:
        prefix.checked(ref, consumed)
    refs = {
        name: {"path": str(ROOT / config[name]), "sha256": config[name + "_sha256"]}
        for name in ("failed_owner", "failed_branch", "failed_model")
    }
    owner, branch, model = (prefix.read(refs[name], consumed) for name in refs)
    prefix.require(
        owner["complete"] is owner["passed"] is False
        and owner["gpu_released"] is True
        and owner["gpu_after"] == {"memory_mib": 0, "compute_apps": []}
        and owner["model_initializations"] == 1
        and owner["model_requests_attempted"] == 5
        and owner["denoise_calls_attempted"] == 175
        and owner["available_branch_ref"] == refs["failed_branch"]
        and owner["available_model_ref"] == refs["failed_model"],
        "The failed owner and released GPU receipt are required",
    )
    lifecycle = prefix.read(owner["workers_ref"], consumed)
    prefix.require(
        not lifecycle["launch_error"]
        and [row["name"] for row in lifecycle["workers"]] == ["simulation", "model"]
        and all(row["actual_wait"] is row["reaped"] is True for row in lifecycle["workers"]),
        "Both original child processes must have been waited for and reaped",
    )
    stream = diagnostic_stream(branch)
    verify_model(inputs, model, branch, consumed)
    derived = {**branch, "runtime": {**branch["runtime"], "continuation": stream}}
    image_root = ROOT / "outputs/causal_stream_live_48_20261111/producer/simulation/runtime"
    chunks, sources = references.decode_actual_references(inputs, derived, image_root=image_root, max_windows=6)
    prefix.require(len(chunks) == len(sources) == 5, "Exactly five actual reference chunks are required")
    prefix.checked(config_ref, consumed)
    prefix.reference(PLAN)
    return {
        "config_ref": config_ref,
        "config": config,
        "refs": refs,
        "live": inputs,
        "branch": branch,
        "stream": stream,
        "chunks": chunks,
        "source_observations": sources,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "model_sha256": inputs["live"]["model_sha256"],
        "source_sha256": inputs["source_sha256"],
    }
