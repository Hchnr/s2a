"""106 independent scalar greedy selection; qualified fit inputs only."""

from collections import Counter
from pathlib import Path

import numpy as np
from arm_wrench_data_independent import attach_current_wrench
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("scripts/arm_wrench_data_independent.py")
    selected = e.json("configs/wrench_coverage_interventions_20261025.json")
    for path, digest in selected["source_sha256"].items():
        e.check(path, digest)
    ref = selected["original_training_config_ref"]
    e.check(ref["path"], ref["sha256"])
    old = e.json(ref["path"])
    assert selected["profiles"] == old["profiles"] and selected["all_new_data_split"] == "fit"
    config = e.json("configs/wrench_conditioned_arm_forward_20261025.json")
    ref = selected["normalization_parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    params = e.json(ref["path"])
    mean = np.asarray(params["state_mean"][83:])
    scale = np.asarray(params["state_scale"][83:])
    assert mean.shape == scale.shape == (6,) and np.all(scale > 0)
    ref = selected["native_data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entries = {r["index"]: r for r in e.json(ref["path"])["trajectories"] if r["split"] == "fit"}
    assert len(entries) == 24
    original = e.json("configs/paired_arm_response_20261025.json")
    memo = {}

    def wrench(index):
        if index not in memo:
            entry = entries[index]
            ref = entry["cache_ref"]
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as cache:
                data = {k: cache[k].copy() for k in ("x_site_pose", "time", "protocol_native_goal_override")}
            memo[index] = attach_current_wrench(data, entry, config, e)
        return memo[index]

    support = []
    for group in (0, 3, 5):
        profile = original["profiles"][group // 2]
        index = next(i for i, r in entries.items() if r["source"] == profile["trajectory_ref"])
        anchor = original["trials"][7 * group]["anchor_cycle"]
        support.append((wrench(index)["x_current_wrench_world"][anchor] - mean) / scale)
    for group, profile in enumerate(old["profiles"]):
        index = profile["native_data_index"]
        assert entries[index]["source"] == profile["trajectory_ref"]
        anchor = old["trials"][7 * group]["anchor_cycle"]
        support.append((wrench(index)["x_current_wrench_world"][anchor] - mean) / scale)
    assert len(support) == 24
    assert len(selected["selection_records"]) == 21 and len(selected["trials"]) == 147
    total = 0
    for group, profile in enumerate(old["profiles"]):
        index = profile["native_data_index"]
        assert profile["trajectory_ref"]["sha256"] not in old["excluded_086_trajectory_sha256"]
        data = wrench(index)
        previous = old["trials"][7 * group]["anchor_cycle"]
        best_score, best_anchor, count = -1.0, None, 0
        for t in range(1, profile["record"]["native_steps"]):
            assert not data["protocol_native_goal_override"][t]
            if t == previous:
                continue
            count += 1
            z = (data["x_current_wrench_world"][t] - mean) / scale
            score = min(float(np.dot(z - point, z - point) / 6) for point in support)
            if score > best_score:
                best_score, best_anchor = score, t
        record = selected["selection_records"][group]
        assert record["profile_index"] == group and record["native_data_index"] == index
        assert record["previous_anchor"] == previous and record["candidate_count"] == count
        assert record["anchor_cycle"] == best_anchor
        np.testing.assert_allclose(
            record["minimum_standardized_wrench_squared_distance"], best_score, rtol=0, atol=1e-10
        )
        value = data["x_current_wrench_world"][best_anchor]
        np.testing.assert_allclose(record["current_wrench_world"], value, rtol=0, atol=1e-10)
        support.append((value - mean) / scale)
        for branch in range(7):
            expected = dict(old["trials"][7 * group + branch])
            expected.update(anchor_cycle=best_anchor, physics_steps=best_anchor + 1, phase="fit_wrench_coverage")
            assert selected["trials"][7 * group + branch] == expected
            total += best_anchor + 1
    assert Counter(p["task"] for p in selected["profiles"]) == {0: 7, 1: 7, 2: 7}
    assert total == selected["physics_steps"] == 13209
    e.recheck()
    root = Path("outputs/wrench_coverage_selection_20261025")
    root.mkdir(exist_ok=False)
    write_once(
        root / "independent_verified.json",
        dict(
            complete=True,
            selection_verified=True,
            anchors=21,
            trials=147,
            planned_control_periods=total,
            physics_steps=0,
            new_fits=0,
            source_sha256=e.hashes,
        ),
    )
    print("106 independent scalar selection verified: 21 anchors /147 branches /13209 planned periods; physics0")


if __name__ == "__main__":
    main()
