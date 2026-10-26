"""132 causal completed-history excitation audit."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/132_causal_history_excitation_20261025.md")
    root = Path("outputs/condition_span_forward_20261025")
    closed = e.json(root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    fit_path = root / "fit_result.json"
    e.check(fit_path, closed["source_sha256"][str(fit_path.resolve())])
    fit = e.json(fit_path)
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    e.json(fit["parameter_ref"]["path"])
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
    excitation = []
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
            ref = native["cache_ref"]
            e.check(ref["path"], ref["sha256"])
            anchor = trial["anchor_cycle"]
            start = max(0, anchor - 20)
            with np.load(ref["path"], allow_pickle=False) as cache:
                commands = cache["action_command"][start:anchor, :3].copy()
            assert len(commands) == anchor - start
            metrics = {}
            for label, values in (
                ("centered", commands - commands.mean(axis=0)),
                ("increments", np.diff(commands, axis=0)),
            ):
                singular = np.linalg.svd(values, compute_uv=False)
                threshold = max(values.shape) * np.finfo(float).eps * singular[0] if len(singular) else 0
                rank = int(np.sum(singular > threshold))
                ratio = float(singular[-1] / singular[0]) if len(singular) == 3 and singular[0] > 0 else 0.0
                metrics[label] = dict(rank=rank, singular_values=singular.tolist(), minimum_maximum_ratio=ratio)
            excitation.append(
                dict(
                    dataset=dataset,
                    group=group,
                    native_index=native["index"],
                    anchor_cycle=anchor,
                    start_cycle=start,
                    completed_periods=len(commands),
                    metrics=metrics,
                )
            )
            states.append(raw[offsets[native["index"]] + trial["anchor_cycle"]])
            targets.append(y[np.asarray(indices) + 4865])
            nominals.append(base[np.asarray(indices) + 4865])
            native_ids.append(native["index"])
            identities.append(dict(dataset=dataset, anchor_group=group, native_index=native["index"]))
    assert len(states) == 66 and len(set(native_ids)) == 24
    assert len(excitation) == 66
    e.recheck()
    output = Path("outputs/causal_history_excitation_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            training_only=True,
            window=20,
            anchors=excitation,
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    for label in ("centered", "increments"):
        print(
            label,
            "rank3",
            sum(r["metrics"][label]["rank"] == 3 for r in excitation),
            "/66",
            "median ratio",
            float(np.median([r["metrics"][label]["minimum_maximum_ratio"] for r in excitation])),
        )
    print("short histories", sum(r["completed_periods"] < 20 for r in excitation))


if __name__ == "__main__":
    main()
