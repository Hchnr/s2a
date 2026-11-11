"""Audit saved-response physical feedback and its independent action replay."""

import argparse
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical

ROOT, RUN = admission.ROOT, admission.OUTPUT
OUTPUT = ROOT / "outputs/quality_rate_repair_recorded_feedback_audit_20261111/result.json"
QA = ROOT / "outputs/qa/quality_rate_repair_recorded_feedback_audit_20261111/result.json"
PRODUCER_SHA = "2c3ab4ea6601fa38bc3db010dc26d3376d7a87ce90fd6098be1f685df512596a"
CHECKER_SHA = "bef1e291dd5c8735c61656c475ea15fb0287aeb91c6509b913a409e8541f55fa"


def source_refs():
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/simulation/runtime/prepare_quality_rate_repair_recorded_feedback.py",
        ROOT / "scripts/simulation/runtime/run_quality_rate_repair_recorded_feedback.py",
        ROOT / "tests/research/native_runtime/test_quality_rate_repair_recorded_audit.py",
    )
    return [prefix.reference(path) for path in paths]


def phase(name, digest, consumed):
    summary = prefix.read({"path": str(RUN / name / "result.json"), "sha256": digest}, consumed)
    prefix.require(
        summary["phase"] == name
        and summary["complete"] is summary["passed"] is summary["sources_unchanged"] is True
        and summary["control_qualified"] is False
        and summary["new_model_initializations"] == summary["new_model_requests"] == summary["new_denoise_calls"] == 0
        and summary["counts"]["control_steps"] == 126
        and summary["counts"]["native_steps"] == 3150
        and summary["counts"]["continuation_steps"] == 46,
        "A completed, bounded saved-reference physical phase is required",
    )
    lifecycle = prefix.read(summary["workers_ref"], consumed)
    workers = lifecycle["workers"]
    prefix.require(
        lifecycle["launch_error"] is None
        and len(workers) == (2 if name == "producer" else 1)
        and all(
            row["actual_wait"] is row["reaped"] is True and row["signal_sent"] is False and row["returncode"] == 0
            for row in workers
        ),
        "All owned physical and relay workers must actually exit and be reaped",
    )
    registration = prefix.read(summary["registration_ref"], consumed)
    for ref in registration["source_refs"]:
        prefix.checked(ref, consumed)
    branch = prefix.read(summary["branch_ref"], consumed)
    prefix.require(branch["complete"] is branch["passed"] is True, "Closed physical branch required")
    return summary, branch


def compare_periods(produced, checked, previous):
    prefix.require(len(produced) == len(checked) == 46 and len(previous) == 45, "Exact saved and replayed periods")
    differences = {key: 0.0 for key in ("state", "q", "qvel", "time")}
    prior_maximum = {key: 0.0 for key in ("state", "command", "q", "qvel", "time")}
    for index, (actual, replayed) in enumerate(zip(produced, checked, strict=True)):
        historical.compare_first_period(actual, replayed)
        differences["state"] = max(
            differences["state"], float(np.max(np.abs(np.asarray(actual["actual_state"]) - replayed["actual_state"])))
        )
        for key in ("q", "qvel", "time"):
            left = np.asarray([item[key] for item in actual["actual_inner"]])
            right = np.asarray([item[key] for item in replayed["actual_inner"]])
            differences[key] = max(differences[key], float(np.max(np.abs(left - right))))
        if index < 45:
            prior = previous[index]
            admission.matched_prior_period(actual, prior)
            for key, field in (("state", "actual_state"), ("command", "command")):
                prior_maximum[key] = max(
                    prior_maximum[key], float(np.max(np.abs(np.asarray(actual[field]) - prior[field])))
                )
            for key in ("q", "qvel", "time"):
                left = np.asarray([item[key] for item in actual["actual_inner"]])
                right = np.asarray([item[key] for item in prior["actual_inner"]])
                prior_maximum[key] = max(prior_maximum[key], float(np.max(np.abs(left - right))))
    return differences, prior_maximum


def verify_references(inputs, producer, relay, consumed):
    runtime = producer["runtime"]
    captures, exchanges = runtime["streaming_captures"], runtime["streaming_model_exchanges"]
    prefix.require(
        len(captures) == len(exchanges) == relay["responses_sent"] == len(relay["records"]) == 6
        and relay["complete"] is relay["passed"] is True
        and relay["new_model_initializations"] == relay["new_model_requests"] == relay["new_denoise_calls"] == 0,
        "Six saved responses and zero new model work are required",
    )
    for index, (capture, exchange, record) in enumerate(zip(captures, exchanges, inputs["records"], strict=True)):
        request = prefix.read(exchange["request_ref"], consumed)
        response = prefix.read(exchange["response_ref"], consumed)
        prefix.checked(capture["canvas_png"], consumed)
        prefix.require(
            request["request_id"] == index
            and request["step"] == record["step"]
            and request["sim_time_s"] == record["sim_time_s"]
            and request["state_sha256"] == record["state_sha256"]
            and request["image"]["sha256"] == response["image_sha256"] == record["image_sha256"]
            and relay["records"][index]["historical_response_sha256"] == record["response_sha256"]
            and response["denoise_calls"] == 35,
            "Every local response must match its original current image, state, clock and historical identity",
        )
        for key in ("action", "normalized_action"):
            np.testing.assert_array_equal(response[key], record[key])


def audit():
    inputs = admission.load_inputs()
    consumed = dict(inputs["source_sha256"])
    producer_summary, producer = phase("producer", PRODUCER_SHA, consumed)
    checker_summary, checker = phase("checker", CHECKER_SHA, consumed)
    relay = prefix.read(producer_summary["relay_ref"], consumed)
    isolation = prefix.read(checker_summary["policy_isolation_ref"], consumed)
    prefix.require(
        isolation["before"] is isolation["after"] is True and checker["candidate_modules_unloaded"] is True,
        "The independent checker must not load the candidate solver",
    )
    verify_references(inputs, producer, relay, consumed)
    produced, checked = admission.periods(producer), admission.periods(checker)
    differences, prior_maximum = compare_periods(produced, checked, inputs["prior_periods"])
    stop = producer["runtime"]["continuation"]["stop"]
    row = produced[-1]
    quality = row["decision"]["quality"]
    prefix.require(
        row["step"] == 125
        and quality["status"] == "model_witness_rate_repaired"
        and quality["verification"]["passed"] is True
        and stop["reason"] == "quality_no_witness"
        and stop["at_step"] == 126
        and stop["next_action_dispatched"] is False
        and checker["runtime"]["continuation"]["reported_producer_stop"] == stop,
        "The newly executed repaired action and subsequent reported stop must match",
    )
    predicted, actual = np.asarray(quality["first_tcp_error"]), np.asarray(row["tcp_error"])
    tail = 0
    for item in reversed(produced):
        if not item["task_success"]:
            break
        tail += 1
    prefix.require(
        tail == 39 and not row["pose_caps_satisfied"] and all(value <= 1e-10 for value in differences.values()),
        "Actual task, pose and replay evidence must retain its original thresholds",
    )
    return {
        "complete": True,
        "passed": True,
        "source_refs": source_refs(),
        "producer_ref": {"path": str(RUN / "producer/result.json"), "sha256": PRODUCER_SHA},
        "checker_ref": {"path": str(RUN / "checker/result.json"), "sha256": CHECKER_SHA},
        "historical_responses": 6,
        "historical_denoise_calls": 210,
        "new_model_requests": 0,
        "new_denoise_calls": 0,
        "physical_periods_per_phase": 126,
        "native_steps_per_phase": 3150,
        "continuation_periods": 46,
        "new_periods_after_previous_stop": 1,
        "prior_period_max_difference": prior_maximum,
        "independent_replay_max_difference": differences,
        "replayed_continuation_inner_samples": 1150,
        "checker_policy_isolated": True,
        "step125": {
            "model_position_error_m": float(np.linalg.norm(predicted[:3])),
            "model_rotation_error_rad": float(np.linalg.norm(predicted[3:])),
            "actual_position_error_m": float(np.linalg.norm(actual[:3])),
            "actual_rotation_error_rad": float(np.linalg.norm(actual[3:])),
            "actual_pose_caps_satisfied": row["pose_caps_satisfied"],
            "task_success": row["task_success"],
            "quality_responses": quality["response_evaluations"],
        },
        "producer_stop": {"reason": stop["reason"], "at_step": stop["at_step"], "next_action_dispatched": False},
        "stop_reason_independently_verified": False,
        "terminal_consecutive_success_periods": tail,
        "minimum_joint_margin_rad": min(item["minimum_inner_margin_rad"] for item in produced),
        "inner_below_buffer_samples": sum(item["inner_below_buffer_samples"] for item in produced),
        "inner_outside_samples": sum(item["inner_outside_samples"] for item in produced),
        "qualified_stable_success": False,
        "control_qualified": False,
        "source_sha256": consumed,
        "automatic_retry": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"],
        "A completed 32-worker audit QA is required",
    )
    prefix.require(
        snapshot(ROOT)["sources"] == prefix.read(qa["source_before_ref"])["sources"],
        "Audit QA and source identity must match",
    )
    print(publish(OUTPUT, audit()), flush=True)


if __name__ == "__main__":
    main()
