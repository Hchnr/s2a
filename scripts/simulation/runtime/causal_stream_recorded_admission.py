"""Source and artifact admission for one recorded-reference stream probe."""

import hashlib
import json

import numpy as np

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/causal_stream_recorded_physical_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/causal_stream_recorded_physical_20261111.md"
OUTPUT = ROOT / "outputs/causal_stream_recorded_physical_20261111"
QA = ROOT / "outputs/qa/causal_stream_recorded_physical_20261111/result.json"
DRY = ROOT / "outputs/causal_stream_recorded_physical_dry_20261111/result.json"
SESSION = "recorded_correct08_seed42_20261111"

FIXED = {
    "schema": "s2a.causal-stream-recorded-physical.v1",
    "output": "outputs/causal_stream_recorded_physical_20261111",
    "historical_config": "configs/tracking/joint_control/causal_wam_two_window_feedback_20261111.json",
    "historical_config_sha256": "8db2a4f09762bbeb6683ec6871d21912b7a3ed827e6b0ade7173bc2bc9ff773c",
    "historical_producer": "outputs/causal_wam_two_window_feedback_20261111/producer/case_08_causal_wam/result.json",
    "historical_producer_sha256": "34c704b3e2dd015ff46d24127b35b115e98599e21d5adf78408de5fade77b233",
    "historical_checker": "outputs/causal_wam_two_window_feedback_20261111/checker/case_08_causal_wam/result.json",
    "historical_checker_sha256": "bea989bafe88e86b9c79f6ea064df8a71e9e3f88cd3d81ba93c605b0b3e7358e",
    "historical_producer_summary": "outputs/causal_wam_two_window_feedback_20261111/producer/result.json",
    "historical_producer_summary_sha256": "9897accc35df0590300cd954f58f6a306eb2d362ae9064cf0e6364804c3928a8",
    "historical_checker_summary": "outputs/causal_wam_two_window_feedback_20261111/checker/result.json",
    "historical_checker_summary_sha256": "4fe32084b7dd08797065096dd84cc29a458b2e9feb9dbc5c050b352667b7658c",
    "historical_checkpoint_manifest": "outputs/wam_independent_20261020/checkpoint_fingerprints.json",
    "historical_checkpoint_manifest_sha256": "624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8",
    "historical_weight_manifest_sha256": "48138bb8c95ce8cc1ca0b70f925e57c8f16bfb21be0a243495b391a06cf5b8ec",
    "case_index": 8,
    "seed": 42,
    "task": "put the cream cheese in the bowl",
    "prefix_steps": 80,
    "window_steps": 8,
    "windows": 2,
    "max_control_steps_per_phase": 96,
    "max_native_steps_per_phase": 2400,
    "max_local_reference_requests": 2,
    "historical_denoise_calls_per_reference": 35,
    "new_model_initializations": 0,
    "new_model_requests": 0,
    "new_denoise_calls": 0,
    "fresh_cases": 0,
    "fits": 0,
    "worker_timeout_seconds": 1800,
    "automatic_retry": False,
}


def read_config():
    prefix.environment()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    return config_ref, config


def validate_config(config):
    prefix.require(
        type(config) is dict
        and set(config) == set(FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in FIXED.items()),
        "The complete recorded-reference physical budget must remain fixed",
    )


def _configured_ref(config, name):
    return {"path": str(ROOT / config[name]), "sha256": config[name + "_sha256"]}


def _check_workers(summary, branch_ref, consumed):
    prefix.require(
        summary["complete"] is summary["passed"] is summary["sources_unchanged"] is True
        and summary["branch_refs"] == [branch_ref],
        "A closed historical phase and its exact branch are required",
    )
    lifecycle = prefix.read(summary["workers_ref"], consumed)
    workers = lifecycle["workers"]
    prefix.require(
        not lifecycle["launch_error"]
        and len(workers) == 1
        and workers[0]["returncode"] == 0
        and workers[0]["actual_wait"] is workers[0]["reaped"] is True
        and workers[0]["signal_sent"] is False,
        "Historical worker must have actually exited and been reaped",
    )


def _fingerprint(manifest, expected):
    matches = [entry for entry in manifest["checkpoints"] if entry["name"] == "8400_masked"]
    prefix.require(len(matches) == 1 and len(matches[0]["files"]) == 163, "Historical 8400 checkpoint roster required")
    files = {path: detail["sha256"] for path, detail in matches[0]["files"].items()}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    prefix.require(digest == expected, "Historical declared weight manifest changed")
    return digest


def load_inputs():
    config_ref, config = read_config()
    inputs = historical.load_inputs()
    consumed = dict(inputs["source_sha256"])
    names = (
        "historical_config",
        "historical_producer",
        "historical_checker",
        "historical_producer_summary",
        "historical_checker_summary",
        "historical_checkpoint_manifest",
    )
    refs = {name: _configured_ref(config, name) for name in names}
    values = {name: prefix.read(ref, consumed) for name, ref in refs.items()}
    producer, checker = values["historical_producer"], values["historical_checker"]
    _check_workers(values["historical_producer_summary"], refs["historical_producer"], consumed)
    _check_workers(values["historical_checker_summary"], refs["historical_checker"], consumed)
    model_sha = _fingerprint(values["historical_checkpoint_manifest"], config["historical_weight_manifest_sha256"])
    for branch in (producer, checker):
        runtime = branch["runtime"]
        prefix.require(
            branch["complete"] is branch["passed"] is runtime["complete"] is runtime["passed"] is True
            and runtime["cleanup_complete"] is True
            and runtime["prefix_control_steps"] == 80
            and runtime["control_steps_completed"] == 96
            and runtime["counters"]["mj_step"]["completed"] == 2400
            and runtime["continuation"]["stop"] is None
            and len(runtime["continuation"]["periods"]) == 16,
            "A complete historical prefix and 16-period feedback or replay are required",
        )
    prefix.require(
        checker["candidate_modules_unloaded"] is True
        and checker["runtime"]["independent_deferral_max_difference"] == 0,
        "Historical checker isolation and gripper proposal agreement required",
    )
    periods = producer["runtime"]["continuation"]["periods"]
    for index, (actual, replayed) in enumerate(
        zip(periods, checker["runtime"]["continuation"]["periods"], strict=True)
    ):
        prefix.require(actual["step"] == replayed["step"] == 80 + index, "Consecutive historical actions required")
        historical.compare_first_period(actual, replayed)

    records = []
    for image, name in zip(inputs["images"], ("response80", "response88"), strict=True):
        response = prefix.read(inputs["refs"][name], consumed)
        arrays = repro.response_arrays(response)
        args, identity = response["args"], response["outputs"][0]["content"]["action128"]
        prefix.require(
            args["prompt"] == config["task"]
            and args["seed"] == identity["seed"] == 42
            and args["num_steps"] == 35
            and args["action_chunk_size"] == 40
            and args["fps"] == identity["action_fps"] == 20
            and identity["checkpoint_path"].endswith("/checkpoints/iter_000008400/model")
            and identity["action_dim_mask"] == [True] * 10 + [False] * 118,
            "Historical response model, task, time and mask must match",
        )
        endpoint = image["current_endpoint"]
        records.append(
            {
                "step": image["frame_index"],
                "sim_time_s": endpoint["sim_time"],
                "state_sha256": endpoint["sim_state_sha256"],
                "image_sha256": image["canvas_png"]["sha256"],
                "response_sha256": inputs["refs"][name]["sha256"],
                "historical_denoise_calls": 35,
                "action": arrays["action"].tolist(),
                "normalized_action": arrays["normalized_action"].tolist(),
            }
        )
    prefix.require([item["step"] for item in records] == [80, 88], "Exactly two historical observations are required")
    prefix.checked(config_ref, consumed)
    prefix.reference(PLAN)
    return {
        "config_ref": config_ref,
        "config": config,
        "refs": refs,
        "old": inputs,
        "producer": producer,
        "checker": checker,
        "periods": periods,
        "records": records,
        "model_sha256": model_sha,
        "source_sha256": consumed,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "refs": inputs["refs"],
        "historical_inputs": historical.input_receipt(inputs["old"]),
        "model_sha256": inputs["model_sha256"],
        "source_sha256": inputs["source_sha256"],
    }


def gate_response(records, request, response):
    index = request.get("request_id")
    prefix.require(type(index) is int and 0 <= index < len(records), "One registered response identity required")
    record = records[index]
    prefix.require(
        request["step"] == record["step"]
        and request["sim_time_s"] == record["sim_time_s"]
        and request["state_sha256"] == record["state_sha256"]
        and request["image"]["sha256"] == response["image_sha256"] == record["image_sha256"],
        "Current observation differs from the saved model request",
    )
    for key in ("action", "normalized_action"):
        prefix.require(np.array_equal(np.asarray(response[key]), np.asarray(record[key])), f"Saved {key} changed")
    return True


def guard_period(periods, period):
    index = period.get("step", -1) - 80
    prefix.require(type(index) is int and 0 <= index < len(periods), "One completed historical period required")
    historical.compare_first_period(period, periods[index])
    return True
