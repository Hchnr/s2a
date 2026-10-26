"""111 fixed kernel response estimates with whole-trajectory exclusion."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/111_local_response_trajectory_leaveout_20261025.md")
    root = Path("outputs/joint_wrench_condition_forward_20261025")
    closed = e.json(root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    fit = e.json(root / "fit_result.json")
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    parameters = e.json(fit["parameter_ref"]["path"])
    with np.load(fit["cache_ref"]["path"], allow_pickle=False) as cache:
        raw, y, base = (cache[k].copy() for k in ("raw", "target", "nominal"))
    assert raw.shape == (4865, 89) and y.shape == base.shape == (5000, 12)
    offsets, offset = {}, 0
    for row in fit["fit_sources"]:
        offsets[row["index"]] = offset
        offset += row["rows"]
    assert offset == 4865
    configs = {
        "086": e.json("configs/paired_arm_response_20261025.json"),
        "097": e.json("configs/training_response_coverage_20261025.json"),
        "106": e.json("configs/wrench_coverage_interventions_20261025.json"),
    }
    config = e.json("configs/joint_wrench_condition_forward_20261025.json")
    ref = config["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entries = e.json(ref["path"])["trajectories"]
    states, targets, nominals, native_ids, identities = [], [], [], [], []
    records = fit["contrast_sources"]
    for dataset in ("086", "097", "106"):
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
    assert len(states) == 45 and len(set(native_ids)) == 24
    z = (np.asarray(states) - parameters["state_mean"]) / parameters["state_scale"]
    targets, nominals, native_ids = np.asarray(targets), np.asarray(nominals), np.asarray(native_ids)
    results = []
    for i in range(45):
        eligible = native_ids != native_ids[i]
        assert np.sum(eligible) in (43, 44)
        distance = np.mean((z[eligible] - z[i]) ** 2, axis=1)
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
    output = Path("outputs/local_response_trajectory_leaveout_20261025")
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
    print("111 training-only whole-trajectory leaveout45anchors; improved over nominal", totals)


if __name__ == "__main__":
    main()
