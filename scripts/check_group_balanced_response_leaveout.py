"""126 group-balanced fixed kernel response estimates with whole-trajectory exclusion."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/126_group_balanced_response_leaveout_20261025.md")
    root = Path("outputs/condition_span_forward_20261025")
    closed = e.json(root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    fit_path = root / "fit_result.json"
    e.check(fit_path, closed["source_sha256"][str(fit_path.resolve())])
    fit = e.json(fit_path)
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    parameters = e.json(fit["parameter_ref"]["path"])
    with np.load(fit["cache_ref"]["path"], allow_pickle=False) as cache:
        raw, y, base = (cache[k].copy() for k in ("raw", "target", "nominal"))
    assert raw.shape == (4865, 89) and y.shape == base.shape == (5063, 12)
    offsets, offset = {}, 0
    for row in fit["fit_sources"]:
        offsets[row["index"]] = offset
        offset += row["rows"]
    assert offset == 4865
    configs = {
        "086": e.json("configs/paired_arm_response_20261025.json"),
        "097": e.json("configs/training_response_coverage_20261025.json"),
        "106": e.json("configs/wrench_coverage_interventions_20261025.json"),
        "115": e.json("configs/condition_span_interventions_20261025.json"),
    }
    config = e.json("configs/condition_span_forward_20261025.json")
    ref = config["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entries = e.json(ref["path"])["trajectories"]
    states, targets, nominals, native_ids, identities = [], [], [], [], []
    records = fit["contrast_sources"]
    for dataset in ("086", "097", "106", "115"):
        groups = sorted({r["anchor_group"] for r in records if r["dataset"] == dataset})
        for group in groups:
            registry = configs[dataset]
            trial = registry["trials"][7 * group]
            profile = registry["profiles"][group // 2 if dataset == "086" else group]
            native = next(r for r in entries if r["source"] == profile["trajectory_ref"])
            assert native["split"] == "fit"
            indices = [i for i, r in enumerate(records) if r["dataset"] == dataset and r["anchor_group"] == group]
            assert [records[i]["axis"] for i in indices] == [0, 1, 2]
            states.append(raw[offsets[native["index"]] + trial["anchor_cycle"]])
            targets.append(y[np.asarray(indices) + 4865])
            nominals.append(base[np.asarray(indices) + 4865])
            native_ids.append(native["index"])
            identities.append(dict(dataset=dataset, anchor_group=group, native_index=native["index"]))
    assert len(states) == 66 and len(set(native_ids)) == 24
    z = (np.asarray(states) - parameters["state_mean"]) / parameters["state_scale"]
    targets, nominals, native_ids = np.asarray(targets), np.asarray(nominals), np.asarray(native_ids)
    results = []
    for i in range(66):
        eligible = native_ids != native_ids[i]
        assert np.sum(eligible) in (63, 65)
        # Fixed physical groups; q sin/cos duplicates intentionally excluded.
        blocks = [
            list(range(7)) + list(range(21, 28)),
            list(range(28, 37)),
            list(range(37, 43)),
            list(range(43, 51)),
            list(range(51, 71)),
            list(range(71, 83)),
            list(range(83, 89)),
        ]
        distance = np.mean([np.mean((z[eligible][:, block] - z[i, block]) ** 2, axis=1) for block in blocks], axis=0)
        weights = (1 + distance) ** -2
        weights /= weights.sum()
        predicted = np.sum(weights[:, None, None] * targets[eligible], axis=0)
        metrics = {}
        for block, name in enumerate(("position", "rotation_vector", "linear_velocity", "angular_velocity")):
            sl = slice(block * 3, block * 3 + 3)
            scale = 1 if block < 2 else 20
            fitted = float(np.linalg.norm((predicted[:, sl] - targets[i, :, sl]) * scale))
            nominal = float(np.linalg.norm((nominals[i, :, sl] - targets[i, :, sl]) * scale))
            metrics[name] = dict(local_error=fitted, nominal_error=nominal, improved=fitted <= nominal)
        results.append(dict(identities[i], excluded_anchor_count=int((~eligible).sum()), metrics=metrics))
    e.recheck()
    output = Path("outputs/group_balanced_response_leaveout_20261025")
    output.mkdir(exist_ok=False)
    totals = {k: sum(r["metrics"][k]["improved"] for r in results) for k in results[0]["metrics"]}
    write_once(
        output / "result.json",
        dict(
            complete=True,
            training_trajectory_leaveout_only=True,
            anchors=results,
            improved_anchor_counts=totals,
            source_sha256=e.hashes,
            parameter_fits=0,
            physics_steps=0,
        ),
    )
    print("old45 query subset", {k: sum(r["metrics"][k]["improved"] for r in results[:45]) for k in totals})
    print("126 training-only whole-trajectory leaveout66anchors; improved over nominal", totals)


if __name__ == "__main__":
    main()
