"""171 time-aligned object departure in audited same-profile case03 logs."""

import json

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    root = REPO / "outputs/object_departure_window_20261025"
    if root.exists():
        raise FileExistsError(root)
    e = Evidence()
    e.read(__file__)
    logs = []
    names = ("causal_residual_loaded_control", "qualified_inverse_enumerated")
    profiles = []
    for name in names:
        profiles.append(e.json(f"configs/{name}_20261025.json")["profiles"][3])
        case = REPO / f"outputs/{name}_20261025/case03"
        assert e.json(case / "independent_verified.json")["mechanical_labels_images_passed"]
        path = next(case.rglob("steps.jsonl"))
        logs.append([json.loads(line) for line in e.read(path).decode().splitlines()])
    assert profiles[0] == profiles[1] and len(logs[0]) == len(logs[1]) == 170
    records = []
    object_name = "akita_black_bowl_1"
    for first, second in zip(*logs, strict=True):
        assert first["step"] == second["step"]
        np.testing.assert_allclose(first["physical_time_after_s"], second["physical_time_after_s"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(first["desired_source"], second["desired_source"], atol=1e-9, rtol=0)
        states = []
        for row in (first, second):
            states.append(
                dict(
                    object_position=row["objects"][object_name]["position"],
                    object_height_gain_m=row["objects"][object_name]["height_gain_m"],
                    after_source=row["after_source"],
                    source_error=row["source_error"],
                    opening_after=row["opening_after"],
                    opening_target=row["opening_target"],
                    command=row["command"],
                    gripper_route=row["gripper"]["route"],
                    success=row["success"],
                )
            )
        object_distance = float(np.linalg.norm(np.asarray(states[0]["object_position"]) - states[1]["object_position"]))
        site_distance = float(
            np.linalg.norm(np.asarray(first["after_source"])[:3, 3] - np.asarray(second["after_source"])[:3, 3])
        )
        records.append(
            dict(
                step=first["step"],
                object_distance_m=object_distance,
                source_position_distance_m=site_distance,
                opening_difference=abs(first["opening_after"] - second["opening_after"]),
                states=states,
            )
        )
    # These are descriptive cutoffs for reporting, not new acceptance thresholds.
    first_object_1cm = next(r["step"] for r in records if r["object_distance_m"] > 0.01)
    first_opening_point05 = next(r["step"] for r in records if r["opening_difference"] > 0.05)
    e.recheck()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            names=names,
            profile=profiles[0],
            object_name=object_name,
            records=records,
            first_object_distance_over_1cm=first_object_1cm,
            first_opening_difference_over_point05=first_opening_point05,
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="Aligned observed divergence only; descriptive cutoffs do not establish contact causality or replace original acceptance gates",
        ),
    )
    print(
        json.dumps(
            dict(
                first_object_distance_over_1cm=first_object_1cm,
                first_opening_difference_over_point05=first_opening_point05,
                window=records[max(0, first_object_1cm - 2) : first_object_1cm + 3],
            )
        )
    )


if __name__ == "__main__":
    main()
