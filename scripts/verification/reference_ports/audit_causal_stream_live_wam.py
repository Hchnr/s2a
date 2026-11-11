"""Independently audit one bounded live WAM stream and physical replay."""

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
from scripts.verification.reference_ports import audit_causal_stream_recorded_physical as recorded_audit
from scripts.verification.reference_ports.audit_wam_feedback_reobserve import validate_ledger
from scripts.wam.tracking.preview import causal_stream_live_admission as admission
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_stream_requests

ROOT = prefix.REPO
RUN = ROOT / "outputs/causal_stream_live_wam_20261111"
PRODUCER_SHA = "200b64e82598fe3e21ae74c8b1cadbd3a2ab818eccb135afdf0c6ad7c7b4b932"
CHECKER_SHA = "824543f2a49ea3a68368bd0f866ac504c238841e8f5506d1a0f377ef8f684833"
PREFLIGHT_SHA = "ce73ea61d4bbd892545141e846769f2ebe876fec9e85226e6cfbf568dee6f9f9"
QA = ROOT / "outputs/qa/causal_stream_live_audit_20261111/result.json"
OUTPUT = ROOT / "outputs/causal_stream_live_audit_20261111/result.json"


def _read(ref, consumed):
    return prefix.read(ref, consumed)


def _phase(name, digest, consumed):
    summary = _read({"path": str(RUN / name / "result.json"), "sha256": digest}, consumed)
    prefix.require(
        summary["phase"] == name
        and summary["complete"] is summary["passed"] is summary["sources_unchanged"] is True
        and summary["gpu_released"] is True
        and summary["counts"]
        == {
            "control_steps": 96,
            "native_steps": 2400,
            "continuation_steps": 16,
            "prefix_inner_samples": 2000,
            "continuation_inner_samples": 400,
        },
        "A fully closed physical phase and released GPU are required",
    )
    lifecycle = _read(summary["workers_ref"], consumed)
    workers = lifecycle["workers"]
    prefix.require(
        lifecycle["launch_error"] is None
        and len(workers) == (2 if name == "producer" else 1)
        and all(
            item["returncode"] == 0 and item["actual_wait"] is item["reaped"] is True and item["signal_sent"] is False
            for item in workers
        ),
        "Every actual worker must exit and be reaped without a signal",
    )
    if name == "producer":
        prefix.require(
            lifecycle["passed"] is True
            and lifecycle["interrupted"] is False
            and summary["gpu_before"] == summary["gpu_after"] == {"memory_mib": 0, "compute_apps": []}
            and summary["model_initializations"] == 1
            and summary["model_requests_attempted"] == 2
            and summary["denoise_calls_attempted"] == 70,
            "Actual GPU use and release must match the registered budget",
        )
    else:
        prefix.require(
            summary["model_initializations"]
            == summary["model_requests_attempted"]
            == summary["denoise_calls_attempted"]
            == 0,
            "Checker must not invoke a model",
        )
    branch = _read(summary["branch_ref"], consumed)
    prefix.require(branch["complete"] is branch["passed"] is True, "Completed physical branch required")
    return summary, branch


def _verify_model(inputs, summary, preflight, consumed):
    model = _read(summary["model_ref"], consumed)
    prefix.require(
        model["complete"] is model["passed"] is True
        and model["gpu_uuid"] == admission.GPU
        and model["model_initialization_attempts"] == model["model_initializations"] == 1
        and model["local_requests_received"]
        == model["model_requests_attempted"]
        == model["model_requests_completed"]
        == model["response_send_attempts"]
        == model["responses_sent"]
        == 2
        and model["denoise_calls_attempted"] == model["denoise_calls_completed"] == 70
        and model["physics_steps"] == model["fresh_cases"] == 0
        and model["automatic_retry"] is False,
        "One initialized WAM model and two actual 35-step requests are required",
    )
    validate_ledger(model["denoise_evidence"], model["mask_evidence"])
    cv2 = model["cv2_admission"]
    probe = _read(preflight["cv2_probe_ref"], consumed)
    prefix.require(
        cv2["complete"] is cv2["ld_restored"] is True
        and cv2["gpu_uuid"] == admission.GPU
        and cv2["sources"] == probe["sources"]
        and cv2["changes"] == probe["changes"]
        and probe["actual_wait"] is probe["reaped"] is True
        and probe["gpu_work"] == 0,
        "The OpenCV process mutation must match its completed CPU probe",
    )
    for path, digest in cv2["sources"].items():
        prefix.require(repro.sha256(path) == digest, "Registered OpenCV source changed")
    imported = model["imported_cosmos_sources"]
    source_count = 0
    for item in imported.values():
        path = item.get("file")
        if path is None:
            continue
        prefix.require(
            path in inputs["model_source_hashes"] and item["sha256"] == inputs["model_source_hashes"][path],
            "An actually imported Cosmos module differs from the frozen source",
        )
        prefix.checked({"path": path, "sha256": item["sha256"]}, consumed)
        source_count += 1
    prefix.require(source_count > 100, "The actual imported Cosmos source roster is incomplete")
    model_runtime_path = RUN / "producer/model/model_run/model/runtime.json"
    model_runtime_ref = prefix.reference(model_runtime_path)
    model_runtime = _read(model_runtime_ref, consumed)
    checkpoint = inputs["prepared"]["model_config"]["checkpoint"]
    expected_checkpoint = (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
    prefix.require(
        Path(model_runtime["checkpoint_path"]).resolve() == expected_checkpoint
        and model_runtime["ema_weights"] is False,
        "Actual loaded checkpoint differs",
    )
    return model, model_runtime_ref, source_count


def verify_model_response(inputs, record, row, exchange, consumed):
    prefix.require(
        row["step"] == record["step"]
        and row["reference_sent"] is True
        and row["historical_response_sha256"] == record["response_sha256"],
        "One completed model response per actual observation is required",
    )
    request = _read(row["model_request_ref"], consumed)
    response = _read(row["actual_response_ref"], consumed)
    live_stream_requests._check_response_identity(
        response,
        model_request=request,
        task=inputs["live_config"]["task"],
        checkpoint=inputs["prepared"]["model_config"]["checkpoint"],
    )
    arrays = repro.response_arrays(response)
    for key, array in arrays.items():
        prefix.require(
            np.array_equal(array, np.asarray(record[key]))
            and row["comparison"][key]["elementwise_equal"] is True
            and row["comparison"][key]["different_elements"] == 0,
            f"Actual live WAM {key} differs from the audited reference",
        )
    directory = Path(row["actual_response_ref"]["path"]).parent
    image = directory / "current.png"
    image_ref = prefix.reference(image)
    prefix.require(
        image_ref["sha256"] == record["image_sha256"] and request["vision_path"] == str(image),
        "The actual model input image differs",
    )
    prefix.checked(image_ref, consumed)
    conditioning_ref = prefix.reference(directory / "conditioning.json")
    conditioning = _read(conditioning_ref, consumed)
    caption = conditioning["caption"]
    caption_sha = hashlib.sha256(json.dumps(caption).encode()).hexdigest()
    prefix.require(
        caption["task_title"] == caption["command"] == inputs["live_config"]["task"]
        and caption["task_plan"] == []
        and conditioning["image_sha256"] == record["image_sha256"]
        and response["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"] == caption_sha,
        "Actual caption and current image must match model response conditioning",
    )
    envelope = _read(exchange["response_ref"], consumed)
    for key, array in arrays.items():
        prefix.require(
            np.array_equal(array, np.asarray(envelope[key])), "Delivered reference differs from actual WAM output"
        )
    return {
        "step": record["step"],
        "image_sha256": record["image_sha256"],
        "actual_response_ref": row["actual_response_ref"],
        "model_request_ref": row["model_request_ref"],
        "conditioning_ref": conditioning_ref,
        "model_elapsed_seconds": row["model_elapsed_seconds"],
        "historical_response_sha256": record["response_sha256"],
    }


def load_audit():
    prefix.environment()
    inputs = admission.load_inputs()
    consumed = dict(inputs["source_sha256"])
    preflight = _read({"path": str(RUN / "preflight.json"), "sha256": PREFLIGHT_SHA}, consumed)
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
        "Complete zero-work model SHA preflight required",
    )
    for ref in preflight["source_refs"]:
        prefix.checked(ref, consumed)
    producer_summary, producer = _phase("producer", PRODUCER_SHA, consumed)
    _checker_summary, checker = _phase("checker", CHECKER_SHA, consumed)
    prefix.require(checker["candidate_modules_unloaded"] is True, "Independent checker isolation required")
    model, runtime_ref, imported_count = _verify_model(inputs, producer_summary, preflight, consumed)
    runtime = producer["runtime"]
    captures, exchanges = runtime["streaming_captures"], runtime["streaming_model_exchanges"]
    prefix.require(len(captures) == len(exchanges) == len(model["responses"]) == 2, "Two current windows required")
    rows = []
    for index, (record, capture, exchange, model_row) in enumerate(
        zip(inputs["records"], captures, exchanges, model["responses"], strict=True)
    ):
        recorded_audit.verify_exchange(
            record, capture, exchange, RUN / "producer/simulation/runtime", consumed, index=index
        )
        rows.append(verify_model_response(inputs, record, model_row, exchange, consumed))
    producer_periods = recorded_audit._periods(producer)
    checker_periods = recorded_audit._periods(checker)
    prefix.require(len(producer_periods) == len(checker_periods) == 16, "Both phases must complete 16 periods")
    max_q = max_v = max_time = 0.0
    for actual, replayed, old in zip(producer_periods, checker_periods, inputs["periods"], strict=True):
        historical.compare_first_period(actual, old)
        historical.compare_first_period(replayed, actual)
        for left, right in zip(actual["actual_inner"], replayed["actual_inner"], strict=True):
            max_q = max(max_q, float(np.max(np.abs(np.asarray(left["q"]) - np.asarray(right["q"])))))
            max_v = max(max_v, float(np.max(np.abs(np.asarray(left["qvel"]) - np.asarray(right["qvel"])))))
            max_time = max(max_time, abs(left["time"] - right["time"]))
    actual_prefix, replay_prefix = runtime["prefix_inner_periods"], checker["runtime"]["prefix_inner_periods"]
    prefix.require(len(actual_prefix) == len(replay_prefix) == 80, "All physical prefix periods required")
    for left, right in zip(actual_prefix, replay_prefix, strict=True):
        common.check_prefix_inner(left, right)
    return {
        "complete": True,
        "passed": True,
        "control_qualified": False,
        "gpu_uuid": admission.GPU,
        "model_initializations": 1,
        "model_requests": 2,
        "denoise_calls": 70,
        "new_audit_model_requests": 0,
        "new_audit_physics_steps": 0,
        "recorded_producer_native_steps": 2400,
        "recorded_checker_native_steps": 2400,
        "replayed_prefix_inner_samples": 2000,
        "replayed_continuation_inner_samples": 400,
        "checker_policy_isolated": True,
        "max_replay_joint_difference_rad": max_q,
        "max_replay_velocity_difference": max_v,
        "max_replay_time_difference_s": max_time,
        "actually_imported_cosmos_source_files": imported_count,
        "model_runtime_ref": runtime_ref,
        "rows": rows,
        **recorded_audit.metrics(producer_periods),
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
