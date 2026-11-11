"""Audit two executed WAM feedback windows and decompose their saved errors."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from s2a.geometry import error_summary
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as study
from scripts.verification.reference_ports import audit_saved_wam_feedback as metrics

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_wam_two_window_metrics_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_wam_two_window_metrics_20261111.md"
ROOT = REPO / "outputs/causal_wam_two_window_metrics_20261111"
QA = REPO / "outputs/qa/causal_wam_two_window_metrics_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            Path(study.__file__),
            Path(metrics.__file__),
            REPO / "tests/research/native_runtime/test_causal_wam_two_window_metrics.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    fixed = {
        "schema": "s2a.causal-wam-two-window-metrics.v1",
        "output": str(ROOT.relative_to(REPO)),
        "steps": 16,
        "window_steps": 8,
        "new_physics_steps": 0,
        "new_model_requests": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    prefix.require(
        all(config.get(key) == value and type(config.get(key)) is type(value) for key, value in fixed.items()),
        "Fixed CPU-only two-window audit required",
    )
    return ref, config


def load_audit():
    config_ref, config = read_config()
    refs = {
        key: study.saved.binding.configured_ref(config, key)
        for key in ("producer", "producer_branch", "checker", "checker_branch", "model_audit")
    }
    producer, branch, checker, checked_branch, model = (
        prefix.read(refs[key]) for key in ("producer", "producer_branch", "checker", "checker_branch", "model_audit")
    )
    prefix.require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
        and checker["complete"] is checker["passed"] is checker["sources_unchanged"] is True
        and producer["branch_refs"] == [refs["producer_branch"]]
        and checker["branch_refs"] == [refs["checker_branch"]]
        and branch["complete"] is branch["passed"] is checked_branch["complete"] is checked_branch["passed"] is True
        and checked_branch["candidate_modules_unloaded"] is True
        and producer["counts"]["continuation_steps"] == checker["counts"]["continuation_steps"] == config["steps"]
        and producer["counts"]["captured_steps"] == checker["counts"]["captured_steps"] == [80, 88]
        and model["complete"] is model["causal_first_reference_exact"] is True,
        "Completed model provenance, physical feedback, and independent replay required",
    )
    registration = prefix.read(producer["registration_ref"])
    prefix.require(registration["source_refs"] == study.source_refs(), "Physical-run source identities changed")
    inputs = study.load_inputs()
    for summary in (producer, checker):
        workers = prefix.read(summary["workers_ref"])["workers"]
        prefix.require(
            len(workers) == 1
            and workers[0]["returncode"] == 0
            and workers[0]["actual_wait"] is workers[0]["reaped"] is True
            and workers[0]["signal_sent"] is False,
            "Waited and reaped physical worker required",
        )
    actual_runtime, replay_runtime = branch["runtime"], checked_branch["runtime"]
    actual, replay = actual_runtime["continuation"]["periods"], replay_runtime["continuation"]["periods"]
    prefix.require(
        len(actual) == len(replay) == 16
        and [p["step"] for p in actual] == [p["step"] for p in replay] == list(range(80, 96))
        and actual_runtime["continuation"]["stop"] is None
        and replay_runtime["continuation"]["stop"] is None,
        "Sixteen matching completed periods required",
    )
    prefix.require(
        len(actual_runtime["prefix_inner_periods"]) == len(replay_runtime["prefix_inner_periods"]) == 80,
        "All eighty independently replayed prefix periods required",
    )
    for source, checked in zip(
        actual_runtime["prefix_inner_periods"], replay_runtime["prefix_inner_periods"], strict=True
    ):
        common.check_prefix_inner(source, checked)
    for index, (source, checked) in enumerate(zip(actual, replay, strict=True)):
        study.compare_first_period(source, checked)
        study.prior.compare_score(source["completed_feedback"]["score"], checked["completed_feedback"]["score"])
        if index < 8:
            study.compare_first_period(source, inputs["periods"][index])
    for expected, source_image, checked_image in zip(
        inputs["images"], actual_runtime["current_wam_images"], replay_runtime["current_wam_images"], strict=True
    ):
        study.validate_live_capture(expected, source_image)
        study.validate_live_capture(expected, checked_image)
    windows = []
    for index in range(2):
        periods = actual[index * 8 : (index + 1) * 8]
        reference = inputs["targets"][index][:8]
        correct = inputs["old"]["base"]["targets"][index][:8]
        reference_opening = inputs["openings"][index][:8]
        correct_opening = inputs["old"]["base"]["openings"][index][:8]
        measured = np.asarray([row["completed_feedback"]["score"]["gripper"]["opening"] for row in periods])
        actual_pose = metrics.actual_from_logged_error(reference, [row["tcp_error"] for row in periods])
        values = metrics.summarize(
            actual_pose, reference, correct, measured, reference_opening, correct_opening, periods
        )
        values["reference_vs_correct"] = error_summary(correct, reference)
        windows.append({"start_step": 80 + index * 8, **values})
    tail_success = 0
    for row in reversed(actual):
        if not row["task_success"]:
            break
        tail_success += 1
    consumed = dict(inputs["source_sha256"])
    for ref in (config_ref, *refs.values(), producer["registration_ref"]):
        prefix.checked(ref, consumed)
    return {
        "complete": True,
        "windows": windows,
        "completed_steps": 16,
        "replayed_prefix_inner_samples": 2000,
        "replayed_continuation_inner_samples": 400,
        "independent_deferral_max_difference": replay_runtime["independent_deferral_max_difference"],
        "terminal_consecutive_success_periods": tail_success,
        "qualified_stable_success": False,
        "new_model_requests": 0,
        "new_physics_steps": 0,
        "fresh_cases": 0,
        "source_sha256": consumed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args()
    prefix.require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("LD_LIBRARY_PATH") == "",
        "CPU-only audit required",
    )
    qa = prefix.read(prefix.reference(QA))
    snapshot = prefix.read(qa["source_before_ref"])
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["skipped_count"] == 0
        and all(
            snapshot["sources"].get(str(Path(ref["path"]).relative_to(REPO))) == ref["sha256"] for ref in source_refs()
        ),
        "Passing CPU audit on current sources required",
    )
    result = load_audit()
    result["qa_ref"] = prefix.reference(QA)
    result["audit_source_refs"] = source_refs()
    ROOT.mkdir(exist_ok=False)
    print(publish(ROOT / "result.json", result), flush=True)
    print(
        json.dumps({key: value for key, value in result.items() if key not in ("source_sha256", "audit_source_refs")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
