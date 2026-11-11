"""Independently audit the recorded-reference stream producer and checker."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.infrastructure.repository_paths import repo_path
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import causal_stream_recorded_admission as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical

ROOT = prefix.REPO
PRODUCER = ROOT / "outputs/causal_stream_recorded_physical_20261111/producer/result.json"
CHECKER = ROOT / "outputs/causal_stream_recorded_physical_20261111/checker/result.json"
PRODUCER_SHA = "43482cc62c975eca919ac51a57d68dbda1c3edc5e9652b1678dc09a0a9acd9e8"
CHECKER_SHA = "4ac014f9f560f06829c8a4f6b2f70f7330a50ba151962f116bb27ebdc1f7e805"
QA = ROOT / "outputs/qa/causal_stream_recorded_audit_20261111/result.json"
OUTPUT = ROOT / "outputs/causal_stream_recorded_audit_20261111/result.json"


def _checked(ref, consumed):
    return prefix.read(ref, consumed)


def _summary(path, digest, phase, consumed):
    value = _checked({"path": str(path), "sha256": digest}, consumed)
    prefix.require(
        value["phase"] == phase
        and value["complete"] is value["passed"] is value["sources_unchanged"] is True
        and value["new_model_initializations"] == value["new_model_requests"] == value["new_denoise_calls"] == 0
        and value["fresh_cases"] == 0
        and value["counts"]
        == {
            "control_steps": 96,
            "native_steps": 2400,
            "continuation_steps": 16,
            "prefix_inner_samples": 2000,
            "continuation_inner_samples": 400,
        },
        "One completed registered CPU phase is required",
    )
    lifecycle = _checked(value["workers_ref"], consumed)
    workers = lifecycle["workers"]
    prefix.require(
        lifecycle["launch_error"] is None
        and len(workers) == (2 if phase == "producer" else 1)
        and all(
            item["returncode"] == 0 and item["actual_wait"] is item["reaped"] is True and item["signal_sent"] is False
            for item in workers
        ),
        "Every actual worker must exit and be reaped without a signal",
    )
    if phase == "producer":
        prefix.require(lifecycle["passed"] is True and lifecycle["interrupted"] is False, "Producer owner failed")
    branch = _checked(value["branch_ref"], consumed)
    prefix.require(branch["complete"] is branch["passed"] is True, "Completed branch required")
    return value, branch


def verify_exchange(record, capture, exchange, runtime_root, consumed, *, index):
    prefix.require(
        capture["frame_index"] == record["step"]
        and capture["current_endpoint"]["sim_state_sha256"] == record["state_sha256"]
        and capture["current_endpoint"]["sim_time"] == record["sim_time_s"]
        and capture["render_before"] == capture["render_after"]
        and capture["render_before"] == {"sim_time": record["sim_time_s"], "sim_state_sha256": record["state_sha256"]}
        and capture["canvas_png"]["sha256"] == record["image_sha256"]
        and capture["future_rgb_frames"] == 0
        and exchange["request_id"] == index
        and all(
            exchange[key] is True
            for key in (
                "request_send_attempted",
                "request_sent",
                "response_receive_attempted",
                "response_received",
                "response_accepted",
                "policy_ready",
            )
        ),
        "The captured current image and completed response exchange must match",
    )
    image = prefix.checked(capture["canvas_png"], consumed)
    prefix.checked(capture["raw_cameras"], consumed)
    request, response = _checked(exchange["request_ref"], consumed), _checked(exchange["response_ref"], consumed)
    relative = Path(request["image"]["path"])
    prefix.require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and repo_path(str((runtime_root / relative).relative_to(ROOT)), root=ROOT) == image
        and request["session"] == admission.SESSION
        and request["request_id"] == index
        and request["seed"] == 42
        and request["model_sha256"] == admission.FIXED["historical_weight_manifest_sha256"]
        and request["denoise_steps"] == response["denoise_calls"] == 35
        and all(
            type(response[key]) is type(request[key]) and response[key] == request[key]
            for key in (
                "version",
                "session",
                "request_id",
                "step",
                "sim_time_s",
                "state_sha256",
                "seed",
                "model_sha256",
            )
        ),
        "The stored request and response identity must match their current capture",
    )
    for key in ("action", "normalized_action"):
        prefix.require(np.array_equal(np.asarray(response[key]), np.asarray(record[key])), f"Saved {key} changed")
    return {
        "step": record["step"],
        "image_sha256": record["image_sha256"],
        "historical_response_sha256": record["response_sha256"],
        "request_ref": exchange["request_ref"],
        "response_ref": exchange["response_ref"],
    }


def _periods(branch):
    return [
        period for row in branch["runtime"]["continuation"]["segments"] for period in row["continuation"]["periods"]
    ]


def metrics(periods):
    prefix.require(len(periods) == 16, "Exactly 16 completed continuation periods required")
    windows = []
    for index in range(2):
        rows = periods[index * 8 : (index + 1) * 8]
        errors = np.asarray([row["tcp_error"] for row in rows], dtype=float)
        prefix.require(errors.shape == (8, 6) and np.isfinite(errors).all(), "Finite TCP errors required")
        windows.append(
            {
                "start_step": rows[0]["step"],
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
    }


def load_audit():
    prefix.environment()
    inputs = admission.load_inputs()
    consumed = dict(inputs["source_sha256"])
    producer_summary, producer = _summary(PRODUCER, PRODUCER_SHA, "producer", consumed)
    _checker_summary, checker = _summary(CHECKER, CHECKER_SHA, "checker", consumed)
    prefix.require(checker["candidate_modules_unloaded"] is True, "Checker policy isolation required")
    relay = _checked(producer_summary["relay_ref"], consumed)
    prefix.require(
        relay["complete"] is relay["passed"] is True
        and relay["kind"] == "saved_response_relay"
        and relay["local_reference_requests"] == relay["responses_sent"] == 2
        and relay["new_model_initializations"] == relay["new_model_requests"] == relay["new_denoise_calls"] == 0
        and [row["historical_response_sha256"] for row in relay["records"]]
        == [row["response_sha256"] for row in inputs["records"]],
        "Both responses must come from the previously audited model artifacts",
    )
    runtime = producer["runtime"]
    captures, exchanges = runtime["streaming_captures"], runtime["streaming_model_exchanges"]
    prefix.require(len(captures) == len(exchanges) == 2, "Two actual captures and exchanges required")
    runtime_root = PRODUCER.parent / "simulation/runtime"
    rows = [
        verify_exchange(record, capture, exchange, runtime_root, consumed, index=index)
        for index, (record, capture, exchange) in enumerate(zip(inputs["records"], captures, exchanges, strict=True))
    ]
    producer_periods, checker_periods = _periods(producer), _periods(checker)
    prefix.require(len(producer_periods) == len(checker_periods) == 16, "Both phases must complete 16 periods")
    max_joint, max_velocity, max_time = 0.0, 0.0, 0.0
    for actual, replayed, old in zip(producer_periods, checker_periods, inputs["periods"], strict=True):
        historical.compare_first_period(actual, old)
        historical.compare_first_period(replayed, actual)
        for left, right in zip(actual["actual_inner"], replayed["actual_inner"], strict=True):
            max_joint = max(max_joint, float(np.max(np.abs(np.asarray(left["q"]) - np.asarray(right["q"])))))
            max_velocity = max(
                max_velocity, float(np.max(np.abs(np.asarray(left["qvel"]) - np.asarray(right["qvel"]))))
            )
            max_time = max(max_time, abs(left["time"] - right["time"]))
    prefix_periods, replay_prefix = runtime["prefix_inner_periods"], checker["runtime"]["prefix_inner_periods"]
    prefix.require(len(prefix_periods) == len(replay_prefix) == 80, "Complete actual prefix samples required")
    for actual, replayed in zip(prefix_periods, replay_prefix, strict=True):
        common.check_prefix_inner(actual, replayed)
    return {
        "complete": True,
        "passed": True,
        "control_qualified": False,
        "new_model_initializations": 0,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "new_audit_physics_steps": 0,
        "recorded_producer_native_steps": 2400,
        "recorded_checker_native_steps": 2400,
        "replayed_prefix_inner_samples": 2000,
        "replayed_continuation_inner_samples": 400,
        "checker_policy_isolated": True,
        "max_replay_joint_difference_rad": max_joint,
        "max_replay_velocity_difference": max_velocity,
        "max_replay_time_difference_s": max_time,
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
