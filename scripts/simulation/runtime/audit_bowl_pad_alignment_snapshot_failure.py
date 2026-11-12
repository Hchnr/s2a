"""Preserve the failed pad-snapshot audit without executing another model."""

import numpy as np

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import audit_bowl_pad_alignment as audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/bowl_pad_alignment_snapshot_failure_20261112/result.json"


def mismatch_summary(points):
    """Compare each recorded pad/body field to its separately copied snapshot."""
    summary = {}
    for key, box_index in (("left_pad_world_m", 0), ("right_pad_world_m", 1)):
        errors = [
            float(
                np.max(
                    np.abs(
                        np.asarray(row["pad_alignment"][key])
                        - np.asarray(row["pad_alignment"]["box_projections"][box_index]["center_world_m"])
                    )
                )
            )
            for row in points
        ]
        summary[key] = dict(
            mismatch_count=sum(error > 1e-12 for error in errors),
            maximum_absolute_difference_m=max(errors),
            reset_absolute_difference_m=errors[0],
            final_absolute_difference_m=errors[-1],
            all_equal_to_final=all(row["pad_alignment"][key] == points[-1]["pad_alignment"][key] for row in points),
        )
    body = [
        float(
            np.max(
                np.abs(np.asarray(row["pad_alignment"]["body_origin_world_m"]) - np.asarray(row["body_origin_world_m"]))
            )
        )
        for row in points
    ]
    summary["body_origin_world_m"] = dict(
        mismatch_count=sum(error > 1e-12 for error in body), maximum_absolute_difference_m=max(body)
    )
    return summary


def main():
    prefix.environment()
    source = prefix.read(audit.SOURCE)
    worker, lifecycle = prefix.read(source["worker_ref"]), prefix.read(source["workers_ref"])
    geometry, counters = worker["geometry"], worker["counters"]
    points = [geometry["reset"], *geometry["points"]["fixed"], *geometry["points"]["lead"]]
    prefix.require(
        source["complete"] is source["passed"] is worker["complete"] is worker["passed"] is True
        and len(points) == 97
        and counters["models"] == counters["data"] == 1
        and counters["forwards_completed"] == 97
        and counters["physics_steps"] == counters["contexts"] == counters["model_requests"] == 0
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True,
        "The completed source work and its actual cost must be preserved",
    )
    try:
        audit.check_point(points[0])
    except AssertionError as error:
        error_type, message = type(error).__name__, str(error)
    else:
        raise ValueError("The prior reset snapshot unexpectedly passed the audit")
    summary = mismatch_summary(points)
    prefix.require(
        summary["left_pad_world_m"]["mismatch_count"] > 0
        and summary["right_pad_world_m"]["mismatch_count"] > 0
        and summary["left_pad_world_m"]["all_equal_to_final"] is True
        and summary["right_pad_world_m"]["all_equal_to_final"] is True,
        "The failed saved snapshots must retain their observed mismatch",
    )
    result = dict(
        complete=True,
        passed=True,
        source_ref=audit.SOURCE,
        worker_ref=source["worker_ref"],
        source_diagnostic_qualified=False,
        first_failed_query="reset",
        error_type=error_type,
        error_message=message,
        mismatch_summary=summary,
        points=97,
        source_models=1,
        source_forwards=97,
        source_physics_steps=0,
        cumulative_static_models=4,
        cumulative_static_forwards=301,
        audit_models=0,
        audit_forwards=0,
        audit_physics_steps=0,
        original_forward_reexecuted=False,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
