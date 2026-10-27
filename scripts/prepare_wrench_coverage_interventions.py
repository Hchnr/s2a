"""106 deterministic farthest-current-wrench anchors from fit inputs only."""

import copy
from pathlib import Path

import numpy as np
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/106_wrench_coverage_interventions_20261025.md")
    config = e.json("configs/wrench_conditioned_arm_forward_20261025.json")
    fit = e.json("outputs/wrench_conditioned_arm_forward_20261025/fit_result.json")
    ref = fit["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    p = e.json(ref["path"])
    mean, scale = np.asarray(p["state_mean"])[83:], np.asarray(p["state_scale"])[83:]
    assert mean.shape == scale.shape == (6,) and np.all(scale > 0)
    ref = config["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entries = [r for r in e.json(ref["path"])["trajectories"] if r["split"] == "fit"]
    assert len(entries) == 24
    old_path = Path("configs/training_response_coverage_20261025.json")
    old = e.json(old_path)
    original = e.json("configs/paired_arm_response_20261025.json")
    assert len(old["profiles"]) == 21 and old["all_new_data_split"] == "fit"
    memo = {}

    def current(entry):
        assert entry["split"] == "fit"
        if entry["index"] not in memo:
            ref = entry["cache_ref"]
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as cache:
                # Selection loads no y labels or heldout observations.
                data = {k: cache[k].copy() for k in ("x_site_pose", "time", "protocol_native_goal_override")}
            data = attach_current_wrench(data, entry, config, e)
            memo[entry["index"]] = data
        return memo[entry["index"]]

    supported = []
    ids = []
    for group in (0, 3, 5):
        profile = original["profiles"][group // 2]
        entry = next(r for r in entries if r["source"]["sha256"] == profile["trajectory_ref"]["sha256"])
        anchor = original["trials"][group * 7]["anchor_cycle"]
        supported.append(current(entry)["x_current_wrench_world"][anchor])
        ids.append(("086", group))
    for group, profile in enumerate(old["profiles"]):
        entry = next(r for r in entries if r["index"] == profile["native_data_index"])
        assert entry["source"] == profile["trajectory_ref"]
        anchor = old["trials"][group * 7]["anchor_cycle"]
        supported.append(current(entry)["x_current_wrench_world"][anchor])
        ids.append(("097", group))
    assert set(ids) == {(r["dataset"], r["anchor_group"]) for r in fit["contrast_sources"]}
    assert len(supported) == 24
    selected = []
    trials = []
    for group, profile in enumerate(old["profiles"]):
        assert profile["trajectory_ref"]["sha256"] not in old["excluded_086_trajectory_sha256"]
        entry = next(r for r in entries if r["index"] == profile["native_data_index"])
        data = current(entry)
        n = len(data["time"])
        original_anchor = old["trials"][group * 7]["anchor_cycle"]
        valid = np.flatnonzero(
            (~data["protocol_native_goal_override"].astype(bool))
            & (np.arange(n) > 0)
            & (np.arange(n) != original_anchor)
        )
        assert len(valid) > 0 and np.all(valid < profile["record"]["native_steps"])
        candidates = (data["x_current_wrench_world"][valid] - mean) / scale
        support = (np.asarray(supported) - mean) / scale
        distances = np.mean((candidates[:, None, :] - support[None, :, :]) ** 2, axis=2).min(axis=1)
        index = int(np.argmax(distances))
        anchor = int(valid[index])
        observed = data["x_current_wrench_world"][anchor].copy()
        selected.append(
            dict(
                profile_index=group,
                native_data_index=entry["index"],
                anchor_cycle=anchor,
                previous_anchor=original_anchor,
                candidate_count=len(valid),
                minimum_standardized_wrench_squared_distance=float(distances[index]),
                current_wrench_world=observed.tolist(),
            )
        )
        supported.append(observed)
        for branch in range(7):
            trial = copy.deepcopy(old["trials"][group * 7 + branch])
            trial.update(anchor_cycle=anchor, physics_steps=anchor + 1, phase="fit_wrench_coverage")
            trials.append(trial)
    assert len(selected) == 21 and len(trials) == 147
    output = copy.deepcopy(old)
    output.update(
        trials=trials,
        physics_steps=sum(r["physics_steps"] for r in trials),
        selection_records=selected,
        selection_rule="per-profile greedy farthest6-current-wrench, fixed104-fit scales; original order; earliest tie",
        original_training_config_ref=dict(path=str(old_path.resolve()), sha256=sha256(old_path)),
        wrench_closure_ref=config["wrench_closure_ref"],
        normalization_parameter_ref=fit["parameter_ref"],
        new_fits=0,
        model_calls=0,
        fresh_cases=0,
    )
    e.recheck()
    output["source_sha256"] = e.hashes
    write_once("configs/wrench_coverage_interventions_20261025.json", output)
    print("106 registered21anchors/147trials/control periods", output["physics_steps"], ";physics0")
    for r in selected:
        print(
            r["native_data_index"],
            r["previous_anchor"],
            "->",
            r["anchor_cycle"],
            r["minimum_standardized_wrench_squared_distance"],
        )


if __name__ == "__main__":
    main()
