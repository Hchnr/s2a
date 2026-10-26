"""105 distinguish native-state support from independently intervened conditions."""

import json
from pathlib import Path

import numpy as np
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence

from s2a.wrench_conditioned_arm_forward import raw_condition, with_history

ROOT = Path("outputs/intervention_condition_coverage_20261025")


def main():
    e = Evidence()
    for p in (
        __file__,
        "plans/v0.0.1/105_intervention_condition_coverage_20261025.md",
        "scripts/arm_wrench_data.py",
        "src/s2a/wrench_conditioned_arm_forward.py",
    ):
        e.read(p)
    config = e.json("configs/wrench_conditioned_arm_forward_20261025.json")
    prior = Path("outputs/wrench_conditioned_arm_forward_20261025")
    closed = e.json(prior / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    fit = e.json(prior / "fit_result.json")
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    parameters = e.json(fit["parameter_ref"]["path"])
    with np.load(fit["cache_ref"]["path"], allow_pickle=False) as cache:
        native_raw = cache["raw"].copy()
    assert native_raw.shape == (4865, 89)
    source = e.json(config["data_result_ref"]["path"])
    original = e.json("configs/paired_arm_response_20261025.json")
    extra = e.json("configs/training_response_coverage_20261025.json")
    offsets = {}
    start = 0
    for r in fit["fit_sources"]:
        offsets[r["index"]] = start
        start += r["rows"]
    assert start == 4865
    memo = {}

    def condition(profile, anchor):
        entry = next(r for r in source["trajectories"] if r["source"]["sha256"] == profile["trajectory_ref"]["sha256"])
        assert entry["split"] == "fit"
        if entry["index"] not in memo:
            ref = entry["cache_ref"]
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as cache:
                data = {k: cache[k].copy() for k in cache.files}
            data = attach_current_wrench(with_history(data, parameters), entry, config, e)
            memo[entry["index"]] = raw_condition(data)
        raw = memo[entry["index"]][anchor]
        fit_row = offsets[entry["index"]] + anchor
        np.testing.assert_allclose(raw, native_raw[fit_row], rtol=0, atol=1e-10)
        return raw, entry["index"], fit_row

    train = []
    for group in (0, 3, 5):
        profile = original["profiles"][group // 2]
        anchor = original["trials"][group * 7]["anchor_cycle"]
        raw, index, fit_row = condition(profile, anchor)
        train.append(
            dict(
                dataset="086",
                anchor_group=group,
                native_index=index,
                task=profile["task"],
                anchor_cycle=anchor,
                native_fit_row=fit_row,
                raw=raw.tolist(),
            )
        )
    for group, profile in enumerate(extra["profiles"]):
        anchor = extra["trials"][group * 7]["anchor_cycle"]
        raw, index, fit_row = condition(profile, anchor)
        train.append(
            dict(
                dataset="097",
                anchor_group=group,
                native_index=index,
                task=profile["task"],
                anchor_cycle=anchor,
                native_fit_row=fit_row,
                raw=raw.tolist(),
            )
        )
    assert len(train) == 24 and len({r["native_index"] for r in train}) == 24
    values = np.array([r["raw"] for r in train])
    mean, scale = np.asarray(parameters["state_mean"]), np.asarray(parameters["state_scale"])
    assert mean.shape == scale.shape == (89,)
    ztrain = (values - mean) / scale
    test = []
    validation = e.json(prior / "validation_result.json")
    for group in range(6):
        profile = original["profiles"][group // 2]
        anchor = original["trials"][group * 7]["anchor_cycle"]
        raw, index, fit_row = condition(profile, anchor)
        z = (raw - mean) / scale
        distance = np.sqrt(np.mean((ztrain - z) ** 2, axis=1))
        force_distance = np.sqrt(np.mean((ztrain[:, 83:] - z[83:]) ** 2, axis=1))
        nearest = int(np.argmin(distance))
        nearest_wrench = int(np.argmin(force_distance))
        same = np.array([r["task"] == profile["task"] for r in train])
        same_values = values[same, 83:]
        result = dict(
            anchor_group=group,
            task=profile["task"],
            native_index=index,
            anchor_cycle=anchor,
            native_fit_row=fit_row,
            native_state_present=True,
            intervention_used_for_fit=group in (0, 3, 5),
            raw=raw.tolist(),
            nearest_condition_training_anchor={k: v for k, v in train[nearest].items() if k != "raw"},
            condition_rms_distance=float(distance[nearest]),
            nearest_wrench_training_anchor={k: v for k, v in train[nearest_wrench].items() if k != "raw"},
            wrench_rms_distance=float(force_distance[nearest_wrench]),
            wrench_axes_outside_training_range=np.flatnonzero(
                (raw[83:] < values[:, 83:].min(axis=0)) | (raw[83:] > values[:, 83:].max(axis=0))
            ).tolist(),
            wrench_axes_outside_same_task_range=np.flatnonzero(
                (raw[83:] < same_values.min(axis=0)) | (raw[83:] > same_values.max(axis=0))
            ).tolist(),
            force_norm=float(np.linalg.norm(raw[83:86])),
            passed_local_gates=int(sum(validation["local_response"][group]["gates"].values())),
        )
        test.append(result)
    e.recheck()
    ROOT.mkdir(exist_ok=False)
    out = dict(
        complete=True,
        training_anchor_count=24,
        training_conditions=train,
        development_conditions=test,
        training_wrench_min=values[:, 83:].min(axis=0).tolist(),
        training_wrench_max=values[:, 83:].max(axis=0).tolist(),
        training_force_norm_max=float(np.linalg.norm(values[:, 83:86], axis=1).max()),
        source_sha256=e.hashes,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
        new_fits=0,
        control_qualification=False,
        scope="read-only marginal ranges/distances; no confidence threshold or new gate",
    )
    with (ROOT / "result.json").open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print("training max force norm", out["training_force_norm_max"])
    for r in test:
        print(
            r["anchor_group"],
            "force",
            r["force_norm"],
            "outside",
            r["wrench_axes_outside_training_range"],
            "same_task_outside",
            r["wrench_axes_outside_same_task_range"],
            "distance",
            r["condition_rms_distance"],
            "wrench_distance",
            r["wrench_rms_distance"],
            "native row",
            r["native_fit_row"],
        )


if __name__ == "__main__":
    main()
