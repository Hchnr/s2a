"""Independently audit a completed seed-43 saved-image model replay."""

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "outputs/audits/adaptive_multiseed_real43_20261113"
PRODUCER = {
    "path": str(ROOT / "outputs/adaptive_multiseed_real43_20261113/producer/result.json"),
    "sha256": "64b018f5fdbd72b3a055e6a60da3194b14522edc321591e5ac291946ab5cad71",
}
PREFLIGHT = {
    "path": str(ROOT / "outputs/adaptive_multiseed_real43_20261113/preflight.json"),
    "sha256": "ee5f52ddf4a80255e2f413f7e0e2073304a7eea90625783efbb9d9cf73e3d4fb",
}
QA = {
    "path": str(ROOT / "outputs/qa/adaptive_multiseed_real43_final_20261113/result.json"),
    "sha256": "876ea392386372c05cbdec65c7c149705a0a01e5c8ba82b9d0d679169c2beb63",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(ref):
    path = Path(ref["path"]).resolve()
    require(path.is_relative_to(ROOT) and sha256(path) == ref["sha256"], "A recorded repository file changed")
    return path


def read(ref):
    return json.loads(checked(ref).read_text())


def arrays(response):
    require(response["status"] == "success" and len(response["outputs"]) == 1, "One successful response is required")
    content = response["outputs"][0]["content"]
    result = {
        "action": np.asarray(content["action"], dtype=np.float64),
        "normalized_action": np.asarray(content["action128"]["normalized_action"], dtype=np.float64),
    }
    require(
        all(
            value.shape == (40, 128) and np.isfinite(value).all() and not np.any(value[:, 10:])
            for value in result.values()
        ),
        "Finite H40 outputs with zero inactive slots are required",
    )
    return result


def compare(left, right):
    """Independently compute whole-array differences and active-element counts."""
    comparison = {}
    for key in ("action", "normalized_action"):
        delta = left[key] - right[key]
        comparison[key] = {
            "elementwise_equal": bool(np.array_equal(left[key], right[key])),
            "max_absolute_difference": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "different_elements": int(np.count_nonzero(delta)),
            "different_active_elements": int(np.count_nonzero(delta[:, :10])),
        }
    return comparison


def verify(
    producer,
    preflight,
    qa,
    registration,
    workers,
    model,
    client,
    actual_request,
    actual_response,
    old_request,
    old_response,
    repeated_response,
    case,
):
    """Validate persisted identities and results without the producer auditor."""
    require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is producer["gpu_released"] is True
        and producer["control_qualified"] is False
        and producer["evidence_level"] == "saved_development_image_model_replay"
        and producer["physics_steps"]
        == producer["native_models"]
        == producer["action_dispatches"]
        == producer["fresh_cases"]
        == 0
        and producer["gpu_before"]["memory_mib"] == producer["gpu_after"]["memory_mib"] == 0
        and producer["gpu_before"]["compute_apps"] == producer["gpu_after"]["compute_apps"] == []
        and producer["gpu_uuid"]
        == preflight["gpu_uuid"]
        == model["gpu_uuid"]
        == "GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf",
        "The completed offline run and released device are required",
    )
    require(
        preflight["complete"] is preflight["passed"] is qa["complete"] is qa["passed"] is qa["all_tests_passed"] is True
        and preflight["source_files_verified"] == 1004
        and preflight["weight_files_verified"] == 163
        and qa["configured_parallelism"] == qa["execution_parallelism"] == 64
        and qa["call_count"] == 185
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and qa["actual_wait"] is qa["reaped"] is qa["source_unchanged"] is True
        and not qa["collection"]["collection_errors"],
        "Preflight and complete CPU regression must match the recorded budgets",
    )
    require(
        workers["passed"] is True
        and workers["launch_error"] is None
        and workers["timed_out"] is workers["interrupted"] is False
        and len(workers["workers"]) == 2
        and {row["name"] for row in workers["workers"]} == {"simulation", "model"}
        and all(
            row["returncode"] == 0 and row["actual_wait"] is row["reaped"] is True and row["signal_sent"] is False
            for row in workers["workers"]
        ),
        "Both owned workers must have actually exited and been reaped",
    )
    require(
        model["complete"] is model["passed"] is client["complete"] is client["passed"] is True
        and model["sampling_seed"] == 43
        and model["model_sequence_complete"] is True
        and model["model_initialization_attempts"] == model["model_initializations"] == 1
        and model["model_requests_attempted"] == model["model_requests_completed"] == model["responses_sent"] == 1
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == len(model["denoise_evidence"]) == 35
        and all(row["completed"] is True for row in model["denoise_evidence"])
        and len(model["mask_evidence"]) == 1
        and model["mask_evidence"][0]["stage"] == "initial_noise"
        and model["mask_evidence"][0]["inactive_rms_after"] == 0.0,
        "The actual model work and inactive-slot mask evidence must be complete",
    )
    exchange = client["exchange"]
    frame, envelope = exchange["request"], exchange["response"]
    initial = registration["inputs"]["recorded_initial"]
    recorded = case["images"][0]
    checkpoint = actual_response["outputs"][0]["content"]["action128"]
    require(
        frame["seed"]
        == envelope["seed"]
        == actual_request["seed"]
        == actual_response["args"]["seed"]
        == checkpoint["seed"]
        == 43
        and frame["task"] == actual_request["prompt"] == "put the bowl on the plate"
        and frame["step"] == initial["step"] == recorded["current_endpoint"]["endpoint_step"] == 0
        and frame["sim_time_s"] == initial["sim_time_s"] == recorded["current_endpoint"]["sim_time"]
        and frame["state_sha256"] == initial["state_sha256"] == recorded["current_endpoint"]["sim_state_sha256"]
        and frame["image"]["sha256"] == envelope["image_sha256"] == recorded["canvas_png"]["sha256"]
        and exchange["snapshot_scope"] == "historical_recorded_endpoint_not_live"
        and exchange["accepted_requests"] == 1
        and exchange["completed_periods"]
        == exchange["physics_steps"]
        == exchange["native_models"]
        == exchange["action_dispatches"]
        == 0
        and checkpoint["action_dim_mask"] == [True] * 10 + [False] * 118
        and all(type(item) is bool for item in checkpoint["action_dim_mask"]),
        "The saved frame, seed and actual model response identity must match",
    )
    identity_keys = ("version", "session", "request_id", "step", "sim_time_s", "state_sha256", "seed", "model_sha256")
    require(
        all(type(frame[key]) is type(envelope[key]) and frame[key] == envelope[key] for key in identity_keys)
        and all(actual_response["args"].get(key) == value for key, value in actual_request.items()),
        "The framed response and actual request arguments must match",
    )
    normalized = copy.deepcopy(actual_request)
    normalized.update(name=old_request["name"], seed=old_request["seed"], vision_path=old_request["vision_path"])
    require(normalized == old_request, "Only seed, request name and copied image path may differ from the old request")
    actual, previous, repeated = arrays(actual_response), arrays(old_response), arrays(repeated_response)
    require(all(np.array_equal(previous[key], repeated[key]) for key in previous), "Saved seed-42 reproduction differs")
    require(
        all(np.array_equal(actual[key], np.asarray(envelope[key], dtype=np.float64)) for key in actual),
        "The model arrays differ from the client envelope",
    )
    comparison = compare(actual, previous)
    for key, values in comparison.items():
        original = producer["audit"]["compared_to_saved_seed42"][key]
        require(
            values["elementwise_equal"] is original["elementwise_equal"]
            and values["different_elements"] == original["different_elements"]
            and abs(values["max_absolute_difference"] - original["max_absolute_difference"]) <= 1e-15
            and abs(values["rmse"] - original["rmse"]) <= 1e-15,
            "Independent array comparison differs from the producer receipt",
        )
    return comparison


def audit():
    producer, preflight, qa = read(PRODUCER), read(PREFLIGHT), read(QA)
    registration = read(producer["registration_ref"])
    workers, model, client = (read(producer[key]) for key in ("workers_ref", "model_ref", "client_ref"))
    row = model["responses"][0]
    actual_request, actual_response = read(row["model_request_ref"]), read(row["actual_response_ref"])
    refs = registration["inputs"]["refs"]
    old_request, old_response, repeated_response, case = (
        read(refs[key])
        for key in ("historical_request", "historical_response", "historical_repeat_response", "historical_case")
    )
    for ref in registration["source_refs"]:
        checked(ref)
    for row in workers["workers"]:
        checked(row["log_ref"])
    checked(refs["historical_image"])
    checked(refs["recorded_canvas"])
    require(
        sha256(actual_request["vision_path"]) == refs["historical_image"]["sha256"]
        and sha256(ROOT / "outputs/adaptive_multiseed_real43_20261113/producer/simulation/current.png")
        == refs["historical_image"]["sha256"],
        "Both copied images must match the saved development image",
    )
    comparison = verify(
        producer,
        preflight,
        qa,
        registration,
        workers,
        model,
        client,
        actual_request,
        actual_response,
        old_request,
        old_response,
        repeated_response,
        case,
    )
    return dict(
        complete=True,
        passed=True,
        evidence_level="independent_saved_model_output_audit",
        control_qualified=False,
        producer_ref=PRODUCER,
        preflight_ref=PREFLIGHT,
        qa_ref=QA,
        audit_source_sha256=sha256(__file__),
        registered_source_files_checked=len(registration["source_refs"]),
        same_image_and_task=True,
        changed_request_fields=["name", "seed", "vision_path"],
        seed=43,
        comparisons=comparison,
        new_model_initializations=0,
        new_model_requests=0,
        new_physics_steps=0,
        fresh_cases=0,
    )


def main():
    require(
        Path.cwd().resolve() == ROOT
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "The independent audit must run CPU-only from the repository",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {"complete": False, "passed": False}
    try:
        result = audit()
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        path = OUTPUT / "result.json"
        with path.open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"path": str(path), "sha256": sha256(path), "passed": result["passed"]}), flush=True)


if __name__ == "__main__":
    main()
