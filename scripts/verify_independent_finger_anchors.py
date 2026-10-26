"""190 independent raw conditions, candidate completeness and novelty ordering."""

import math

import numpy as np
from arm_joint_wrench_independent import augment, conditions
from arm_wrench_data_independent import attach_current_wrench
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish


def main():
    out = REPO / "outputs/independent_finger_anchor_selection_20261025/independent_verified.json"
    if out.exists():
        raise FileExistsError(out)
    e = Evidence()
    e.read(__file__)
    original = e.json("outputs/independent_finger_anchor_selection_20261025/result.json")
    for p, h in original["source_sha256"].items():
        e.check(p, h)
    config = e.json("configs/left_table_forward_20261025.json")
    parameters = e.json("outputs/left_table_fit_20261025/parameters.json")
    source = e.json(config["data_result_ref"]["path"])
    used = {}
    profiles = {}
    for name in (
        "training_response_coverage",
        "wrench_coverage_interventions",
        "condition_span_interventions",
        "left_table_interventions",
    ):
        registry = e.json(f"configs/{name}_20261025.json")
        for index, profile in enumerate(registry["profiles"]):
            native = profile["native_data_index"]
            used.setdefault(native, set()).add(registry["trials"][7 * index]["anchor_cycle"])
            profiles.setdefault(native, profile)
    points = np.asarray(original["training_points"])
    candidates = {}
    for entry in source["trajectories"]:
        native = entry["index"]
        if entry["split"] != "fit" or native in (0, 10, 20):
            continue
        assert profiles[native]["trajectory_ref"]["sha256"] == entry["source"]["sha256"]
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            data = augment({k: a[k].copy() for k in a.files}, parameters)
        data = attach_current_wrench(data, entry, config, e)
        raw = conditions(data)
        for cycle, row in enumerate(raw):
            if cycle in used[native]:
                continue
            gap = row[43] - row[44]
            target = 0.04 + 0.02 * (row[47] - row[48])
            if target > 0.005 or gap < 0.005:
                continue
            z = (row - parameters["state_mean"]) / parameters["state_scale"]
            distance = min(math.sqrt(sum(float(v) * float(v) for v in point - z)) for point in points)
            candidates[(native, cycle)] = distance
    recorded = {(r["native_index"], r["anchor_cycle"]): r for r in original["candidates"]}
    assert set(candidates) == set(recorded)
    for key, score in candidates.items():
        np.testing.assert_allclose(score, recorded[key]["distance"], atol=1e-9, rtol=0)
    ranked = sorted(candidates, key=lambda k: (-candidates[k], k[0], k[1]))
    selected = []
    seen = set()
    for key in ranked:
        if key[0] in seen:
            continue
        seen.add(key[0])
        selected.append(key)
        if len(selected) == 3:
            break
    assert selected == [(r["native_index"], r["anchor_cycle"]) for r in original["selected"]]
    assert sum(7 * (cycle + 1) for _, cycle in selected) == original["planned_physics_steps"]
    e.recheck()
    publish(
        out,
        dict(
            complete=True,
            independent_selection_verified=True,
            candidate_count=len(candidates),
            selected=selected,
            planned_physics_steps=original["planned_physics_steps"],
            source_sha256=e.hashes,
            physics_steps=0,
            fits=0,
        ),
    )
    print("190 independent candidate/score/order verified", selected)


if __name__ == "__main__":
    main()
