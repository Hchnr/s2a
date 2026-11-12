"""Check saved bowl geometry calculations without constructing another model."""

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import diagnose_reset_bowl_saved_geometry as geometry
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/reset_bowl_saved_geometry_recovery_audit_20261112/result.json"
SOURCE = {
    "path": str(ROOT / "outputs/reset_bowl_saved_geometry_recovery_20261112/result.json"),
    "sha256": "7690a98a146d49b446167390c4eddac412c79470b0413428d55f5c4e97cd08f3",
}
FAILURE = {
    "path": str(ROOT / "outputs/reset_bowl_saved_geometry_20261112/result.json"),
    "sha256": "18ea6ee0293f0dcb627eacfefb3673699be21fc9b0623b0ce09367836faaec57",
}


def check_point(point):
    """Recompute reported origin distances and validate change-accounting metadata."""
    tcp = np.asarray(point["tcp_world_m"], dtype=float)
    body = np.asarray(point["body_origin_world_m"], dtype=float)
    delta = tcp - body
    prefix.require(
        tcp.shape == body.shape == (3,) and np.isfinite([*tcp, *body]).all(), "Finite world positions are required"
    )
    np.testing.assert_allclose(point["tcp_minus_body_world_m"], delta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(point["body_origin_distance_m"], np.linalg.norm(delta), atol=1e-12, rtol=0)
    np.testing.assert_allclose(point["body_origin_xy_distance_m"], np.linalg.norm(delta[:2]), atol=1e-12, rtol=0)
    np.testing.assert_allclose(point["tcp_above_body_origin_m"], delta[2], atol=1e-12, rtol=0)
    change = point["state_change"]
    spans = {index for quaternion in change["quaternion_norms"] for index in quaternion["state_indices"]}
    prefix.require(
        set(change["changed_indices"]) <= spans and change["max_absolute_change"] <= 1e-12,
        "Only recorded machine-precision quaternion changes are permitted",
    )
    for quaternion in change["quaternion_norms"]:
        if quaternion["max_absolute_change"]:
            prefix.require(
                quaternion["max_absolute_change"] <= 1e-12
                and abs(quaternion["norm_before"] - 1) <= 1e-12
                and abs(quaternion["norm_after"] - 1) <= 1e-12,
                "Changed quaternion norms must remain within the registered bound",
            )


def main():
    prefix.environment()
    source, failure = prefix.read(SOURCE), prefix.read(FAILURE)
    worker, lifecycle = prefix.read(source["worker_ref"]), prefix.read(source["workers_ref"])
    failed_worker = prefix.read(failure["available_worker_ref"])
    prefix.require(
        source["complete"] is source["passed"] is source["sources_unchanged"] is True
        and worker["complete"] is worker["passed"] is True
        and failure["complete"] is failure["passed"] is False
        and source["automatic_retry"] is failure["automatic_retry"] is False
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["signal_sent"] is False,
        "The completed and failed diagnostics and worker lifecycle must remain available",
    )
    counters, failed = worker["counters"], failed_worker["counters"]
    prefix.require(
        counters["models"] == counters["data"] == failed["models"] == failed["data"] == 1
        and counters["forwards_attempted"] == counters["forwards_completed"] == 97
        and failed["forwards_attempted"] == failed["forwards_completed"] == 10
        and counters["physics_steps"] == counters["contexts"] == counters["model_requests"] == 0
        and failed["physics_steps"] == failed["contexts"] == failed["model_requests"] == 0,
        "The actual cumulative geometry budget must be retained",
    )
    value = worker["geometry"]
    reset, points, checks = value["reset"], value["points"], value["capture_checks"]
    prefix.require(
        set(points) == {"fixed", "reobserved"}
        and len(points["fixed"]) == len(points["reobserved"]) == 48
        and len(checks) == 8
        and all(row["max_absolute_difference_m"] <= 1e-9 for row in checks),
        "All saved endpoints and independent TCP anchor checks are required",
    )
    all_points = [reset, *points["fixed"], *points["reobserved"]]
    for point in all_points:
        check_point(point)
    for left, right in zip(points["fixed"][:16], points["reobserved"][:16], strict=True):
        np.testing.assert_array_equal(left["tcp_world_m"], right["tcp_world_m"])
        np.testing.assert_array_equal(left["body_origin_world_m"], right["body_origin_world_m"])
    summaries = {key: geometry.summarize_points(rows, reset["body_origin_world_m"]) for key, rows in points.items()}
    prefix.require(
        summaries == value["summaries"] == source["summaries"],
        "Reported branch geometry must be reproducible from saved points",
    )
    changed = [point["state_change"] for point in all_points if point["state_change"]["changed_indices"]]
    normalization = value["normalization_summary"]
    prefix.require(
        normalization["queries"] == len(all_points) == 97
        and normalization["queries_with_change"] == len(changed)
        and normalization["max_absolute_change"]
        == max(point["state_change"]["max_absolute_change"] for point in all_points)
        and normalization["changed_state_indices"]
        == sorted({index for point in changed for index in point["changed_indices"]}),
        "Every observed normalization change must be accounted for",
    )
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        source_ref=SOURCE,
        failure_ref=FAILURE,
        worker_ref=source["worker_ref"],
        reset=reset,
        summaries=summaries,
        normalization_summary=normalization,
        capture_checks=checks,
        original_forward_reexecuted=False,
        cumulative_models=2,
        cumulative_data=2,
        cumulative_forwards=107,
        cumulative_physics_steps=0,
        cumulative_contexts=0,
        cumulative_model_requests=0,
        audit_models=0,
        audit_forwards=0,
        automatic_retry=False,
    )
    print(publish(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
