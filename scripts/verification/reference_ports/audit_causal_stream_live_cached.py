"""Audit six real WAM requests and the independent cached-stream replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.verification.reference_ports import audit_causal_stream_recorded_physical as recorded
from scripts.wam.tracking.preview import causal_stream_live_cached_admission as admission
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_stream_requests
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine
from scripts.wam.tracking.preview import run_causal_stream_live_cached as owner

ROOT = prefix.REPO
RUN = ROOT / "outputs/causal_stream_live_cached_20261111"
QA = ROOT / "outputs/qa/causal_stream_live_cached_audit_20261111/result.json"
OUTPUT = ROOT / "outputs/causal_stream_live_cached_audit_20261111/result.json"
PRODUCER_SHA = "ccbad44dbbc18529e970faf6e690e939e040dc883325999837d5c461a9cc36cb"
CHECKER_SHA = "538bd325659c8a0def257f88f5737f7e991219a73697732cf2a1fa9c2ef35cea"
PREFLIGHT_SHA = "e66af5d270204b950b496ca315dd48fe6ae985eaf31b6452169a830a00d8ed07"


def phase(name, digest, consumed):
    summary = prefix.read({"path": str(RUN / name / "result.json"), "sha256": digest}, consumed)
    expected = {
        "control_steps": 125,
        "native_steps": 3125,
        "continuation_steps": 45,
        "prefix_inner_samples": 2000,
        "continuation_inner_samples": 1125,
    }
    prefix.require(
        summary["phase"] == name
        and summary["complete"] is summary["passed"] is summary["sources_unchanged"] is True
        and summary["gpu_released"] is True
        and summary["counts"] == expected,
        "A closed 80+45-period phase and released GPU are required",
    )
    lifecycle = prefix.read(summary["workers_ref"], consumed)
    workers = lifecycle["workers"]
    prefix.require(
        lifecycle["launch_error"] is None
        and len(workers) == (2 if name == "producer" else 1)
        and all(
            row["returncode"] == 0 and row["actual_wait"] is row["reaped"] is True and row["signal_sent"] is False
            for row in workers
        ),
        "All registered workers must exit and be reaped without a signal",
    )
    if name == "producer":
        prefix.require(
            lifecycle["passed"] is True
            and lifecycle["interrupted"] is False
            and summary["gpu_before"] == summary["gpu_after"] == {"memory_mib": 0, "compute_apps": []}
            and summary["model_initializations"] == 1
            and summary["model_requests_attempted"] == 6
            and summary["denoise_calls_attempted"] == 210,
            "The actual GPU work and release must match registration",
        )
    else:
        prefix.require(
            summary["model_initializations"]
            == summary["model_requests_attempted"]
            == summary["denoise_calls_attempted"]
            == 0,
            "The independent checker must not run a model",
        )
    branch = prefix.read(summary["branch_ref"], consumed)
    prefix.require(branch["complete"] is branch["passed"] is True, "A closed physical branch is required")
    return summary, branch


def verify_model(inputs, summary, preflight, consumed):
    model = prefix.read(summary["model_ref"], consumed)
    prefix.require(
        model["complete"] is model["passed"] is model["model_sequence_complete"] is True
        and model["model_sequence_stop"] is None
        and model["gpu_uuid"] == admission.GPU
        and model["model_initialization_attempts"] == model["model_initializations"] == 1
        and model["local_requests_received"]
        == model["model_requests_attempted"]
        == model["model_requests_completed"]
        == model["response_send_attempts"]
        == model["responses_sent"]
        == 6
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 210
        and model["physics_steps"] == model["fresh_cases"] == 0
        and model["automatic_retry"] is False,
        "One initialized model and six actual requests are required",
    )
    ledger = model["denoise_evidence"]
    labels = [f"s2a_correct08_t{step}_s42" for step in (80, 88, 96, 104, 112, 120) for _ in range(35)]
    prefix.require(
        len(ledger) == 210
        and [row["index"] for row in ledger] == list(range(210))
        and [row["label"] for row in ledger] == labels
        and all(row["completed"] is True for row in ledger),
        "All 210 actual denoise calls must complete in request order",
    )
    engine.validate_mask_evidence(model["mask_evidence"], 6)
    cv2, probe = model["cv2_admission"], prefix.read(preflight["cv2_probe_ref"], consumed)
    prefix.require(
        cv2["complete"] is cv2["ld_restored"] is True
        and cv2["gpu_uuid"] == admission.GPU
        and cv2["sources"] == probe["sources"]
        and cv2["changes"] == probe["changes"]
        and probe["actual_wait"] is probe["reaped"] is True
        and probe["gpu_work"] == 0,
        "OpenCV environment changes must match the completed CPU probe",
    )
    for path, digest in cv2["sources"].items():
        prefix.require(repro.sha256(path) == digest, "Registered OpenCV source changed")
    imported_count = 0
    for item in model["imported_cosmos_sources"].values():
        path = item.get("file")
        if path is not None:
            prefix.require(
                path in inputs["model_source_hashes"] and item["sha256"] == inputs["model_source_hashes"][path],
                "An imported Cosmos module differs from its frozen source",
            )
            prefix.checked({"path": path, "sha256": item["sha256"]}, consumed)
            imported_count += 1
    prefix.require(imported_count > 100, "The actual Cosmos source roster is incomplete")
    runtime_ref = prefix.reference(RUN / "producer/model/model_run/model/runtime.json")
    runtime = prefix.read(runtime_ref, consumed)
    checkpoint = inputs["prepared"]["model_config"]["checkpoint"]
    expected = (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
    prefix.require(
        Path(runtime["checkpoint_path"]).resolve() == expected and runtime["ema_weights"] is False,
        "Checkpoint identity",
    )
    return model, runtime_ref, imported_count


def verify_response(inputs, capture, exchange, row, *, index, consumed):
    step = 80 + index * 8
    request = prefix.read(row["model_request_ref"], consumed)
    response = prefix.read(row["actual_response_ref"], consumed)
    delivered = prefix.read(exchange["response_ref"], consumed)
    live_stream_requests._check_response_identity(
        response,
        model_request=request,
        task=inputs["live_config"]["task"],
        checkpoint=inputs["prepared"]["model_config"]["checkpoint"],
    )
    arrays = repro.response_arrays(response)
    prefix.require(
        row["step"] == capture["frame_index"] == exchange["step"] == step
        and row["reference_sent"] is True
        and all(np.array_equal(value, np.asarray(delivered[key])) for key, value in arrays.items()),
        "The delivered reference must equal the actual model response",
    )
    if index < 2:
        record = inputs["records"][index]
        prefix.require(
            row["historical_response_sha256"] == record["response_sha256"]
            and all(np.array_equal(value, np.asarray(record[key])) for key, value in arrays.items())
            and all(item["elementwise_equal"] is True for item in row["comparison"].values()),
            "The first two current responses must match the audited history",
        )
    else:
        prefix.require(
            row["historical_response_sha256"] is row["comparison"] is None, "Later responses must be newly generated"
        )
        record = {
            "step": step,
            "state_sha256": capture["current_endpoint"]["sim_state_sha256"],
            "sim_time_s": capture["current_endpoint"]["sim_time"],
            "image_sha256": capture["canvas_png"]["sha256"],
            "response_sha256": row["actual_response_ref"]["sha256"],
            **{key: value.tolist() for key, value in arrays.items()},
        }
    recorded.verify_exchange(record, capture, exchange, RUN / "producer/simulation/runtime", consumed, index=index)
    directory = Path(row["actual_response_ref"]["path"]).parent
    image_ref = prefix.reference(directory / "current.png")
    prefix.require(
        image_ref["sha256"] == capture["canvas_png"]["sha256"] and request["vision_path"] == image_ref["path"],
        "The actual model conditioning must use the current captured image",
    )
    prefix.checked(image_ref, consumed)
    conditioning_ref = prefix.reference(directory / "conditioning.json")
    conditioning = prefix.read(conditioning_ref, consumed)
    caption = conditioning["caption"]
    caption_sha = hashlib.sha256(json.dumps(caption).encode()).hexdigest()
    prefix.require(
        caption["task_title"] == caption["command"] == inputs["live_config"]["task"]
        and caption["task_plan"] == []
        and conditioning["image_sha256"] == capture["canvas_png"]["sha256"]
        and response["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"] == caption_sha,
        "The actual caption and image must match model conditioning",
    )
    return {"step": step, "image_sha256": image_ref["sha256"], "model_elapsed_seconds": row["model_elapsed_seconds"]}


def metrics(periods):
    prefix.require(len(periods) == 45, "Exactly 45 completed feedback periods are required")
    windows = []
    for offset in range(0, len(periods), 8):
        rows = periods[offset : offset + 8]
        errors = np.asarray([row["tcp_error"] for row in rows], dtype=float)
        prefix.require(errors.shape == (len(rows), 6) and np.isfinite(errors).all(), "Finite TCP errors required")
        windows.append(
            {
                "start_step": rows[0]["step"],
                "periods": len(rows),
                "position_rmse_m": float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                "rotation_rmse_rad": float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                "pose_caps_satisfied": sum(row["pose_caps_satisfied"] for row in rows),
                "target_grasp_periods": sum(row["target_grasp"] for row in rows),
                "task_success_periods": sum(row["task_success"] for row in rows),
            }
        )
    trailing = 0
    for row in reversed(periods):
        if not row["task_success"]:
            break
        trailing += 1
    return {
        "windows": windows,
        "terminal_consecutive_success_periods": trailing,
        "qualified_stable_success": trailing >= 40,
        "minimum_joint_margin_rad": min(row["minimum_inner_margin_rad"] for row in periods),
        "inner_below_buffer_samples": sum(row["inner_below_buffer_samples"] for row in periods),
        "inner_outside_samples": sum(row["inner_outside_samples"] for row in periods),
    }


def load_audit():
    prefix.environment()
    inputs = admission.load_inputs()
    consumed = dict(inputs["source_sha256"])
    preflight = prefix.read({"path": str(RUN / "preflight.json"), "sha256": PREFLIGHT_SHA}, consumed)
    prefix.require(
        preflight["complete"] is preflight["passed"] is True
        and preflight["gpu_uuid"] == admission.GPU
        and preflight["source_files_verified"] == 1004
        and preflight["weight_files_verified"] == 163
        and preflight["model_initializations"]
        == preflight["model_requests"]
        == preflight["denoise_calls"]
        == preflight["physics_steps"]
        == 0,
        "Complete zero-work model preflight is required",
    )
    for ref in preflight["source_refs"]:
        prefix.checked(ref, consumed)
    producer_summary, producer = phase("producer", PRODUCER_SHA, consumed)
    _checker_summary, checker = phase("checker", CHECKER_SHA, consumed)
    prefix.require(checker["candidate_modules_unloaded"] is True, "Checker policy isolation required")
    model, model_runtime_ref, imported_count = verify_model(inputs, producer_summary, preflight, consumed)
    runtime = producer["runtime"]
    captures, exchanges = runtime["streaming_captures"], runtime["streaming_model_exchanges"]
    prefix.require(len(captures) == len(exchanges) == len(model["responses"]) == 6, "Six actual exchanges are required")
    rows = [
        verify_response(inputs, capture, exchange, row, index=index, consumed=consumed)
        for index, (capture, exchange, row) in enumerate(zip(captures, exchanges, model["responses"], strict=True))
    ]
    producer_periods, checker_periods = recorded._periods(producer), recorded._periods(checker)
    prefix.require(len(producer_periods) == len(checker_periods) == 45, "Both phases must complete 45 actions")
    maximum = {"state": 0.0, "q": 0.0, "qvel": 0.0, "time": 0.0}
    for index, (actual, replayed) in enumerate(zip(producer_periods, checker_periods, strict=True)):
        historical.compare_first_period(replayed, actual)
        if index < 16:
            historical.compare_first_period(actual, inputs["periods"][index])
        maximum["state"] = max(
            maximum["state"], float(np.max(np.abs(np.asarray(actual["actual_state"]) - replayed["actual_state"])))
        )
        for key in ("q", "qvel", "time"):
            left = np.asarray([row[key] for row in actual["actual_inner"]])
            right = np.asarray([row[key] for row in replayed["actual_inner"]])
            maximum[key] = max(maximum[key], float(np.max(np.abs(left - right))))
    actual_prefix, checker_prefix = runtime["prefix_inner_periods"], checker["runtime"]["prefix_inner_periods"]
    prefix.require(len(actual_prefix) == len(checker_prefix) == 80, "All prefix samples are required")
    for left, right in zip(actual_prefix, checker_prefix, strict=True):
        common.check_prefix_inner(left, right)
    producer_cache, checker_cache = owner.check_cache_events(producer), owner.check_cache_events(checker)
    prefix.require(
        producer_cache == {"observed": 45, "reused": [88, 96, 104, 112, 120]}
        and checker_cache == {"observed": 45, "reused": [88, 96, 104, 112, 120, 125]},
        "Actual producer and checker endpoint reuse must match their completed actions",
    )
    stream, replay = runtime["continuation"], checker["runtime"]["continuation"]
    stop = stream["stop"]
    prefix.require(
        stream["complete"] is False
        and stop["reason"] == "quality_no_witness"
        and stop["solver_status"] == "nonlinear_search_no_witness"
        and stop["at_step"] == 125
        and stop["next_action_dispatched"] is False
        and replay["reported_producer_stop"] == stop
        and replay["stop_reason_independently_verified"] is False,
        "The producer's pre-dispatch quality stop must remain explicitly unverified by replay",
    )
    return {
        "complete": True,
        "passed": True,
        "control_qualified": False,
        "gpu_uuid": admission.GPU,
        "model_initializations": 1,
        "model_requests": 6,
        "denoise_calls": 210,
        "new_audit_model_requests": 0,
        "new_audit_physics_steps": 0,
        "recorded_producer_native_steps": 3125,
        "recorded_checker_native_steps": 3125,
        "replayed_prefix_inner_samples": 2000,
        "replayed_continuation_inner_samples": 1125,
        "checker_policy_isolated": True,
        "replay_max_difference": maximum,
        "actually_imported_cosmos_source_files": imported_count,
        "model_runtime_ref": model_runtime_ref,
        "producer_stop": {"reason": stop["reason"], "solver_status": stop["solver_status"], "at_step": 125},
        "stop_reason_independently_verified": False,
        "producer_cache": producer_cache,
        "checker_cache": checker_cache,
        "rows": rows,
        **metrics(producer_periods),
        "source_sha256": consumed,
        "automatic_retry": False,
    }


def execute():
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"],
        "Completed source-frozen audit tests required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Audit QA source identity changed")
    result = load_audit()
    prefix.require(snapshot(ROOT)["sources"] == before, "Audit sources changed")
    result["qa_ref"] = prefix.reference(QA)
    result["audit_source_ref"] = prefix.reference(Path(__file__).resolve())
    print(publish(OUTPUT, result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    execute()


if __name__ == "__main__":
    main()
