"""Recompute the reset-origin correct-reference comparison from saved receipts."""

from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.references.oracle_timeline import OracleTimeline
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as replay_audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as experiment

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/reset_correct_reference_20261116/audit.json"
PRODUCER = {
    "path": str(ROOT / "outputs/reset_correct_reference_20261116/producer/result.json"),
    "sha256": "7236573a008090168f619c8055c2c8a7af6efd0e51be6a427fa69a2a6e41dd4f",
}
CHECKER = {
    "path": str(ROOT / "outputs/reset_correct_reference_20261116/checker/result.json"),
    "sha256": "f6270a34d14f9339ca94dd2b50c1d48dc0c89822ad45a5edada9b5085477f511",
}


def audit_targets(timeline, rows):
    """Check every original TCP and opening target, including the final window."""
    if not isinstance(timeline, OracleTimeline) or len(rows) != timeline.periods:
        raise ValueError("The original rows must cover the registered oracle timeline")
    pose_max = opening_max = 0.0
    for step, row in enumerate(rows):
        if type(row.get("step")) is not int or row["step"] != step:
            raise ValueError("Original target rows must be in consecutive period order")
        poses, openings, _ = timeline.window(step)
        expected_pose = np.asarray(row["desired_tcp"], dtype=float)
        expected_opening = float(row["gripper"]["preview"]["original_reference"])
        if expected_pose.shape != (4, 4) or not np.isfinite(expected_pose).all() or not np.isfinite(expected_opening):
            raise ValueError("Finite original TCP and opening targets are required")
        pose_max = max(pose_max, float(np.max(np.abs(expected_pose - poses[0]))))
        opening_max = max(opening_max, abs(expected_opening - float(openings[0])))
    if pose_max > 1e-10 or opening_max > 1e-10:
        raise ValueError("The original and reset-origin target timelines differ")
    return dict(periods=timeline.periods, tcp_max_difference=pose_max, opening_max_difference=opening_max)


def audit_branches(producer, checker, rows, timeline):
    """Recompute target parity, replay parity, error metrics, and arm timing."""
    experiment.validate_branch(producer, "producer")
    experiment.validate_branch(checker, "checker")
    left, right = experiment.periods(producer), experiment.periods(checker)
    target_parity = audit_targets(timeline, rows)
    if len(left) != timeline.periods:
        raise ValueError("This comparison requires the full original trajectory")
    replay_maxima = replay_audit.compare_periods(left, right)
    for step, period in enumerate(left):
        reference = timeline.window(step)[1][0]
        if abs(float(period["completed_feedback"]["score"]["opening_reference"]) - reference) > 1e-10:
            raise ValueError("The recorded producer opening target differs from the oracle timeline")
    errors = np.asarray([row["tcp_error"] for row in left], dtype=float)
    opening = np.asarray([row["completed_feedback"]["score"]["opening_error"] for row in left], dtype=float)
    baseline_errors = np.asarray([row["tcp_error"] for row in rows], dtype=float)
    baseline_opening = np.asarray(
        [row["opening"] - row["gripper"]["preview"]["original_reference"] for row in rows], dtype=float
    )
    seconds = np.asarray([row["decision"]["decision_seconds"] for row in left], dtype=float)
    if (
        errors.shape != (timeline.periods, 6)
        or baseline_errors.shape != errors.shape
        or not all(np.isfinite(value).all() for value in (errors, opening, baseline_errors, baseline_opening, seconds))
        or np.any(seconds <= 0)
    ):
        raise ValueError("Finite period errors and positive measured decision times are required")

    def rms(value):
        return float(np.sqrt(np.mean(value**2)))

    def pose_rms(value):
        return rms(np.linalg.norm(value[:, :3], axis=1)), rms(np.linalg.norm(value[:, 3:], axis=1))

    position, rotation = pose_rms(errors)
    old_position, old_rotation = pose_rms(baseline_errors)
    summary = producer["runtime"]["continuation"]["summary"]
    np.testing.assert_allclose(
        [position, rotation, rms(opening)],
        [summary["position_rmse_m"], summary["rotation_rmse_rad"], summary["opening_rmse"]],
        atol=1e-12,
        rtol=0,
    )
    return dict(
        target_parity=target_parity,
        replay_max_difference=replay_maxima,
        baseline=dict(
            position_rmse_m=old_position,
            rotation_rmse_rad=old_rotation,
            opening_rmse=rms(baseline_opening),
            stable_last_40=all(row["success"] is True for row in rows[-40:]),
        ),
        producer=dict(
            position_rmse_m=position,
            rotation_rmse_rad=rotation,
            opening_rmse=rms(opening),
            position_max_m=float(np.max(np.linalg.norm(errors[:, :3], axis=1))),
            rotation_max_rad=float(np.max(np.linalg.norm(errors[:, 3:], axis=1))),
            opening_max_abs=float(np.max(np.abs(opening))),
            stable_last_40=all(row["task_success"] is True for row in left[-40:]),
        ),
        arm_preview_refiner_timing=dict(
            samples=len(seconds),
            median_seconds=float(np.median(seconds)),
            p95_seconds=float(np.percentile(seconds, 95)),
            maximum_seconds=float(np.max(seconds)),
            above_50ms=int(np.sum(seconds > 0.05)),
        ),
    )


def main():
    prefix.environment()
    producer, checker = prefix.read(PRODUCER), prefix.read(CHECKER)
    for group in (producer, checker):
        workers = group["lifecycle"]["workers"]
        prefix.require(
            group["complete"] is group["passed"] is group["sources_unchanged"] is True
            and group["world_model_calls"] == group["fresh_cases"] == 0
            and len(workers) == 1
            and workers[0]["returncode"] == 0
            and workers[0]["actual_wait"] is workers[0]["reaped"] is True,
            "Both physical owners and workers must be closed and source-bound",
        )
    registrations = [prefix.read(group["registration_ref"]) for group in (producer, checker)]
    prefix.require(
        registrations[0]["identity"] == registrations[1]["identity"], "Both phases must use identical inputs"
    )
    identity = registrations[0]["identity"]
    baseline = prefix.read(identity["baseline_ref"])
    baseline_check = prefix.read(identity["baseline_checker_ref"])
    prefix.require(
        baseline["complete"] is baseline["passed"] is baseline["performance_qualified"] is True
        and baseline_check["complete"] is baseline_check["passed"] is True
        and baseline_check["result_ref"] == identity["baseline_ref"],
        "The original baseline and its independent check are required",
    )
    rows = prefix.read(identity["scoring_rows_ref"])
    timeline = OracleTimeline(prefix.read(identity["reference_ref"]), periods=baseline["physics_steps"])
    metrics = audit_branches(prefix.read(producer["branch_ref"]), prefix.read(checker["branch_ref"]), rows, timeline)
    np.testing.assert_allclose(
        [metrics["baseline"]["position_rmse_m"], metrics["baseline"]["rotation_rmse_rad"]],
        [baseline["position_rmse_m"], baseline["so3_rmse_rad"]],
        atol=1e-12,
        rtol=0,
    )
    print(
        publish_result(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                producer_ref=PRODUCER,
                checker_ref=CHECKER,
                baseline_ref=identity["baseline_ref"],
                baseline_checker_ref=identity["baseline_checker_ref"],
                reference_ref=identity["reference_ref"],
                scoring_rows_ref=identity["scoring_rows_ref"],
                auditor_ref=prefix.reference(Path(__file__).resolve()),
                metrics=metrics,
                world_model_calls=0,
                fresh_cases=0,
                physical_steps=0,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
