"""Decompose independently replayed physical tracking errors for one WAM window."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import error_summary, pose_error, validate_transform
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as common
from scripts.simulation.runtime import run_saved_wam_feedback as study

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/saved_wam_feedback_metrics_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/saved_wam_feedback_metrics_20261111.md"
ROOT = REPO / "outputs/saved_wam_feedback_metrics_20261111"
QA = REPO / "outputs/qa/saved_wam_feedback_metrics_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            Path(study.__file__),
            REPO / "src/s2a/geometry/__init__.py",
            REPO / "tests/research/native_runtime/test_saved_wam_feedback_metrics.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    fixed = {
        "schema": "s2a.saved-wam-feedback-metrics.v1",
        "output": str(ROOT.relative_to(REPO)),
        "steps": 8,
        "new_physics_steps": 0,
        "new_model_requests": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    prefix.require(
        all(config.get(key) == value and type(config.get(key)) is type(value) for key, value in fixed.items()),
        "One fixed CPU-only tracking audit required",
    )
    return ref, config


def actual_from_logged_error(targets, errors):
    targets = validate_transform(np.asarray(targets, dtype=float))
    errors = np.asarray(errors, dtype=float)
    prefix.require(
        targets.ndim == 3
        and targets.shape[1:] == (4, 4)
        and errors.shape == (len(targets), 6)
        and np.isfinite(errors).all(),
        "Finite paired TCP targets and logged errors required",
    )
    actual = targets.copy()
    actual[:, :3, 3] = targets[:, :3, 3] - errors[:, :3]
    error_rotation = Rotation.from_rotvec(errors[:, 3:]).as_matrix()
    actual[:, :3, :3] = error_rotation.swapaxes(-1, -2) @ targets[:, :3, :3]
    actual = validate_transform(actual)
    np.testing.assert_allclose(pose_error(targets, actual), errors, atol=1e-10, rtol=0)
    return actual


def summarize(actual, reference, correct, measured_opening, reference_opening, correct_opening, periods):
    reference_error = pose_error(reference, actual)
    correct_error = pose_error(correct, actual)
    reference_opening_error = measured_opening - reference_opening
    correct_opening_error = measured_opening - correct_opening
    within = (np.linalg.norm(reference_error[:, :3], axis=1) <= 0.01) & (
        np.linalg.norm(reference_error[:, 3:], axis=1) <= 0.05
    )
    np.testing.assert_array_equal(within, [period["pose_caps_satisfied"] for period in periods])
    correct_within = (np.linalg.norm(correct_error[:, :3], axis=1) <= 0.01) & (
        np.linalg.norm(correct_error[:, 3:], axis=1) <= 0.05
    )
    return {
        "actual_vs_reference": error_summary(reference, actual),
        "actual_vs_correct": error_summary(correct, actual),
        "opening_vs_reference_rmse": float(np.sqrt(np.mean(reference_opening_error**2))),
        "opening_vs_correct_rmse": float(np.sqrt(np.mean(correct_opening_error**2))),
        "pose_caps_against_reference": int(np.sum(within)),
        "pose_caps_against_correct": int(np.sum(correct_within)),
        "target_grasp_periods": sum(bool(period["target_grasp"]) for period in periods),
        "task_success_periods": sum(bool(period["task_success"]) for period in periods),
        "minimum_inner_margin_rad": min(float(period["minimum_inner_margin_rad"]) for period in periods),
        "inner_outside_samples": sum(int(period["inner_outside_samples"]) for period in periods),
    }


def load_audit():
    config_ref, config = read_config()
    refs = {
        key: study.binding.configured_ref(config, key)
        for key in ("producer", "producer_branch", "checker", "checker_branch", "correct_checker")
    }
    producer, branch, checker, checked_branch, baseline_checker = (
        prefix.read(refs[key])
        for key in ("producer", "producer_branch", "checker", "checker_branch", "correct_checker")
    )
    prefix.require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
        and checker["complete"] is checker["passed"] is checker["sources_unchanged"] is True
        and producer["branch_refs"] == [refs["producer_branch"]]
        and checker["branch_refs"] == [refs["checker_branch"]]
        and branch["complete"] is branch["passed"] is checked_branch["complete"] is checked_branch["passed"] is True
        and checked_branch["candidate_modules_unloaded"] is True
        and baseline_checker["complete"] is baseline_checker["passed"] is True
        and producer["counts"]["continuation_steps"] == checker["counts"]["continuation_steps"] == config["steps"]
        and baseline_checker["counts"]["continuation_steps"] == 16,
        "Completed actual WAM feedback and both independent replay results required",
    )
    registration = prefix.read(producer["registration_ref"])
    prefix.require(
        registration["source_refs"] == study.source_refs(), "Original physical-run source identities changed"
    )
    inputs = study.load_inputs()
    for summary, result in ((producer, branch), (checker, checked_branch)):
        workers = prefix.read(summary["workers_ref"])["workers"]
        prefix.require(
            len(workers) == 1
            and workers[0]["returncode"] == 0
            and workers[0]["actual_wait"] is workers[0]["reaped"] is True
            and workers[0]["signal_sent"] is False,
            "A waited, reaped worker is required",
        )
        study.validate_live_capture(inputs["source_capture"], result["runtime"]["current_wam_image"])
    n = config["steps"]
    recorded = branch["runtime"]["continuation"]["periods"]
    replayed = checked_branch["runtime"]["continuation"]["periods"]
    baseline = inputs["correct"]["runtime"]["continuation"]["periods"][:n]
    prefix.require(
        [p["step"] for p in recorded]
        == [p["step"] for p in replayed]
        == [p["step"] for p in baseline]
        == list(range(80, 80 + n)),
        "Matching ordered actual periods required",
    )
    actual_prefix = branch["runtime"]["prefix_inner_periods"]
    replayed_prefix = checked_branch["runtime"]["prefix_inner_periods"]
    prefix.require(len(actual_prefix) == len(replayed_prefix) == 80, "Every replayed prefix period required")
    for actual, replay in zip(actual_prefix, replayed_prefix, strict=True):
        common.check_prefix_inner(actual, replay)
    for actual, replay in zip(recorded, replayed, strict=True):
        for key in ("command", "actual_state", "tcp_error"):
            np.testing.assert_allclose(actual[key], replay[key], atol=1e-10, rtol=0)
        for key in ("q", "qvel", "time"):
            np.testing.assert_allclose(
                [x[key] for x in actual["actual_inner"]], [x[key] for x in replay["actual_inner"]], atol=1e-10, rtol=0
            )
        prefix.require(
            len(actual["actual_inner"]) == len(replay["actual_inner"]) == 25,
            "Every 25-sample actual period must be replayed",
        )
    wam, correct = inputs["targets"][:n], inputs["base"]["targets"][0][:n]
    wam_opening, correct_opening = inputs["openings"][:n], inputs["base"]["openings"][0][:n]
    wam_actual = actual_from_logged_error(wam, [p["tcp_error"] for p in recorded])
    baseline_actual = actual_from_logged_error(correct, [p["tcp_error"] for p in baseline])
    measured = np.asarray([p["completed_feedback"]["score"]["gripper"]["opening"] for p in recorded])
    baseline_measured = np.asarray([p["completed_feedback"]["score"]["gripper"]["opening"] for p in baseline])
    consumed = dict(inputs["source_sha256"])
    for ref in (config_ref, *refs.values(), producer["registration_ref"]):
        prefix.checked(ref, consumed)
    return {
        "complete": True,
        "steps": n,
        "actual_wam": summarize(wam_actual, wam, correct, measured, wam_opening, correct_opening, recorded),
        "actual_correct_baseline": summarize(
            baseline_actual, correct, correct, baseline_measured, correct_opening, correct_opening, baseline
        ),
        "wam_reference_vs_correct": error_summary(correct, wam),
        "replayed_prefix_inner_samples": 2000,
        "replayed_continuation_inner_samples": n * 25,
        "independent_deferral_max_difference": checked_branch["runtime"]["independent_deferral_max_difference"],
        "original_wam_worker_success": False,
        "initial_noise_ledger_available": False,
        "new_physics_steps": 0,
        "new_model_requests": 0,
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
        "Passing CPU audit on the current sources required",
    )
    result = load_audit()
    result["qa_ref"] = prefix.reference(QA)
    result["audit_source_refs"] = source_refs()
    ROOT.mkdir(exist_ok=False)
    print(publish(ROOT / "result.json", result), flush=True)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_sha256", "audit_source_refs")}), flush=True)


if __name__ == "__main__":
    main()
