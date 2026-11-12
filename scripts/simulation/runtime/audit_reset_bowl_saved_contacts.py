"""Independently check saved contact receipts without running MuJoCo again."""

import collections
import math

import numpy as np

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
SOURCE = {
    "path": str(ROOT / "outputs/reset_bowl_saved_contacts_20261112/result.json"),
    "sha256": "c904ffc6a94d97986c1c1d59c8d9803cf0a74d3f8436ff0169be5557fa1d4912",
}
PRIOR = {
    "path": str(ROOT / "outputs/reset_bowl_saved_geometry_recovery_20261112/result.json"),
    "sha256": "7690a98a146d49b446167390c4eddac412c79470b0413428d55f5c4e97cd08f3",
}
OUTPUT = ROOT / "outputs/reset_bowl_saved_contacts_audit_20261112/result.json"
TARGET = {f"akita_black_bowl_1_g{i}" for i in range(1, 41)}
PADS = {"left_pad": "gripper0_finger1_pad_collision", "right_pad": "gripper0_finger2_pad_collision"}


def check_contacts(value):
    """Check contact categories and native bilateral semantics independently."""
    prefix.require(set(value) == {"left_pad", "right_pad", "other", "bilateral"}, "Contact categories must be exact")
    for category in ("left_pad", "right_pad", "other"):
        for pair in value[category]:
            prefix.require(
                set(pair) == {"target_geom", "other_geom", "distance_m"}
                and pair["target_geom"] in TARGET
                and isinstance(pair["other_geom"], str)
                and bool(pair["other_geom"])
                and math.isfinite(pair["distance_m"]),
                "Each contact pair requires a registered target and finite signed distance",
            )
            prefix.require(
                pair["other_geom"] == PADS[category] if category in PADS else pair["other_geom"] not in PADS.values(),
                "Pad contacts must appear in the correct category",
            )
    prefix.require(
        type(value["bilateral"]) is bool and value["bilateral"] == bool(value["left_pad"] and value["right_pad"]),
        "The bilateral flag must reproduce the native contact-list predicate",
    )


def summarize(rows):
    prefix.require(
        len(rows) == 48 and [row["endpoint_step"] for row in rows] == list(range(1, 49)),
        "Exactly 48 ordered endpoints are required",
    )
    counts = {key: 0 for key in ("left_pad", "right_pad", "bilateral", "other")}
    mismatches = []
    other = collections.defaultdict(list)
    for row in rows:
        value = row["contacts"]
        check_contacts(value)
        for key in ("left_pad", "right_pad", "other"):
            counts[key] += bool(value[key])
        counts["bilateral"] += value["bilateral"]
        prefix.require(type(row["target_grasp"]) is bool, "Original grasp labels must be Boolean")
        if value["bilateral"] != row["target_grasp"]:
            mismatches.append(row["endpoint_step"])
        for pair in value["other"]:
            other[pair["other_geom"]].append((row["endpoint_step"], pair["target_geom"], pair["distance_m"]))
    others = {
        name: dict(
            endpoint_steps=sorted({step for step, _, _ in pairs}),
            contact_points=len(pairs),
            target_geoms=sorted({target for _, target, _ in pairs}),
            minimum_distance_m=min(distance for _, _, distance in pairs),
            maximum_distance_m=max(distance for _, _, distance in pairs),
        )
        for name, pairs in sorted(other.items())
    }
    return dict(endpoint_counts=counts, grasp_label_mismatch_steps=mismatches), others


def main():
    prefix.environment()
    source, prior = prefix.read(SOURCE), prefix.read(PRIOR)
    worker, old_worker = prefix.read(source["worker_ref"]), prefix.read(prior["worker_ref"])
    lifecycle = prefix.read(source["workers_ref"])
    counters = worker["counters"]
    prefix.require(
        source["complete"] is source["passed"] is source["sources_unchanged"] is True
        and source["automatic_retry"] is False
        and worker["complete"] is worker["passed"] is True
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["signal_sent"] is False
        and counters["models"] == counters["data"] == 1
        and counters["forwards_attempted"] == counters["forwards_completed"] == 97
        and counters["physics_steps"] == counters["contexts"] == counters["model_requests"] == 0,
        "The saved result, exact work and reaped worker are required",
    )
    geometry, old = worker["geometry"], old_worker["geometry"]
    check_contacts(geometry["reset"]["contacts"])
    summaries, other_geoms = {}, {}
    prefix.require(set(geometry["points"]) == {"fixed", "reobserved"}, "The two registered branches are required")
    for branch, rows in geometry["points"].items():
        summary, others = summarize(rows)
        prefix.require(summary == geometry["contact_summaries"][branch], "Contact summaries must match all endpoints")
        for point, previous in zip(rows, old["points"][branch], strict=True):
            prefix.require(
                all(point[key] == previous[key] for key in ("state_sha256", "target_grasp", "task_success")),
                "Original endpoint identity and labels must match the earlier geometry result",
            )
            for name in ("tcp_world_m", "body_origin_world_m"):
                np.testing.assert_allclose(point[name], previous[name], atol=1e-12, rtol=0)
        summaries[branch], other_geoms[branch] = summary, others
    for left, right in zip(geometry["points"]["fixed"][:16], geometry["points"]["reobserved"][:16], strict=True):
        prefix.require(left["contacts"] == right["contacts"], "Shared saved-prefix contact lists must match")
    prefix.require(geometry["prior_geometry_checks"] == 97, "All original points must have been compared")
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        source_ref=SOURCE,
        prior_ref=PRIOR,
        worker_ref=source["worker_ref"],
        summaries=summaries,
        other_geoms=other_geoms,
        shared_prefix_endpoints=16,
        audit_models=0,
        audit_forwards=0,
        audit_physics_steps=0,
        original_forward_reexecuted=False,
        cumulative_models=3,
        cumulative_data=3,
        cumulative_forwards=204,
        cumulative_physics_steps=0,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
