"""115 select training-only directions outside the existing condition span."""

import copy
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/115_condition_span_interventions_20261025.md")
    root = Path("outputs/joint_wrench_condition_forward_20261025")
    closed = e.json(root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    fit = e.json(root / "fit_result.json")
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    p = e.json(fit["parameter_ref"]["path"])
    with np.load(fit["cache_ref"]["path"], allow_pickle=False) as cache:
        raw = cache["raw"].copy()
    assert raw.shape == (4865, 89)
    z = (raw - p["state_mean"]) / p["state_scale"]
    phase = z[:, :83] @ np.asarray(p["frequencies"]).T
    joint = z @ np.asarray(p["joint_frequencies"]).T
    phi = np.c_[
        np.ones(len(z)),
        z[:, :83],
        np.cos(phase) / 8,
        np.sin(phase) / 8,
        z[:, 83:],
        np.cos(joint) / np.sqrt(32),
        np.sin(joint) / np.sqrt(32),
    ]
    assert phi.shape == (4865, 282)
    offsets, start = {}, 0
    for row in fit["fit_sources"]:
        offsets[row["index"]] = start
        start += row["rows"]
    original = e.json("configs/paired_arm_response_20261025.json")
    old = e.json("configs/training_response_coverage_20261025.json")
    recent_path = Path("configs/wrench_coverage_interventions_20261025.json")
    recent = e.json(recent_path)
    config = e.json("configs/joint_wrench_condition_forward_20261025.json")
    ref = config["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entries = [r for r in e.json(ref["path"])["trajectories"] if r["split"] == "fit"]
    support, identities = [], []
    for dataset, registry in (("086", original), ("097", old), ("106", recent)):
        groups = (0, 3, 5) if dataset == "086" else range(21)
        for group in groups:
            profile = registry["profiles"][group // 2 if dataset == "086" else group]
            entry = next(r for r in entries if r["source"] == profile["trajectory_ref"])
            anchor = registry["trials"][group * 7]["anchor_cycle"]
            support.append(phi[offsets[entry["index"]] + anchor])
            identities.append((dataset, group))
    assert set(identities) == {(r["dataset"], r["anchor_group"]) for r in fit["contrast_sources"]}
    assert len(support) == 45
    selected, trials = [], []
    for group, profile in enumerate(old["profiles"]):
        index = profile["native_data_index"]
        entry = next(r for r in entries if r["index"] == index)
        assert entry["source"] == profile["trajectory_ref"]
        assert profile["trajectory_ref"]["sha256"] not in old["excluded_086_trajectory_sha256"]
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            override = cache["protocol_native_goal_override"].astype(bool)
        excluded = {registry["trials"][group * 7]["anchor_cycle"] for registry in (old, recent)}
        candidates = np.array([t for t in range(1, profile["record"]["native_steps"]) if t not in excluded])
        assert len(candidates) and not override[candidates].any()
        matrix = np.asarray(support)
        _, singular, right = np.linalg.svd(matrix, full_matrices=False)
        rank = int(np.sum(singular > max(matrix.shape) * np.finfo(float).eps * singular[0]))
        basis = right[:rank]
        values = phi[offsets[index] + candidates]
        residual = values - (values @ basis.T) @ basis
        scores = np.sum(residual**2, axis=1) / np.sum(values**2, axis=1)
        chosen = int(np.argmax(scores))
        anchor = int(candidates[chosen])
        support.append(values[chosen].copy())
        selected.append(
            dict(
                profile_index=group,
                native_data_index=index,
                anchor_cycle=anchor,
                excluded_anchors=sorted(excluded),
                previous_support_rank=rank,
                candidate_count=len(candidates),
                unspanned_feature_squared_fraction=float(scores[chosen]),
            )
        )
        for branch in range(7):
            trial = copy.deepcopy(recent["trials"][group * 7 + branch])
            trial.update(anchor_cycle=anchor, physics_steps=anchor + 1, phase="fit_condition_span")
            trials.append(trial)
    output = copy.deepcopy(recent)
    output.update(
        trials=trials,
        physics_steps=sum(t["physics_steps"] for t in trials),
        selection_records=selected,
        selection_rule="per-profile maximum normalized orthogonal condition-feature residual; earliest tie",
        original_training_config_ref=dict(path=str(recent_path.resolve()), sha256=sha256(recent_path)),
        normalization_parameter_ref=fit["parameter_ref"],
        selection_fit_cache_ref=fit["cache_ref"],
    )
    e.recheck()
    output["source_sha256"] = e.hashes
    write_once("configs/condition_span_interventions_20261025.json", output)
    print("115 selected21anchors/147branches;planned periods", output["physics_steps"], ";physics0")
    print(
        [
            (r["native_data_index"], r["anchor_cycle"], round(r["unspanned_feature_squared_fraction"], 4))
            for r in selected
        ]
    )


if __name__ == "__main__":
    main()
