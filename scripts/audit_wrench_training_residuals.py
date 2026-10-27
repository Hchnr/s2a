"""108 read-only decomposition of the qualified107 fit equations."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/108_wrench_training_residual_audit_20261025.md")
    root = Path("outputs/wrench_coverage_forward_20261025")
    closed = e.json(root / "closure_verified.json")
    assert closed["complete"] and closed["numerically_verified"] and closed["owner_normal_wait_verified"]
    fit = e.json(root / "fit_result.json")
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    params = e.json(fit["parameter_ref"]["path"])
    with np.load(fit["cache_ref"]["path"], allow_pickle=False) as cache:
        x, y, nominal = (cache[k].copy() for k in ("features", "target", "nominal"))
        groups, row_kind = (cache[k].copy() for k in ("groups", "row_kind"))
    assert x.shape == (5000, 872) and y.shape == nominal.shape == (5000, 12)
    assert np.count_nonzero(row_kind) == 135
    pred = nominal + x @ np.asarray(params["coefficients"])
    channels = ("position", "rotation_vector", "linear_velocity", "angular_velocity")

    def summary(mask):
        result = {}
        for i, name in enumerate(channels):
            scale = 1 if i < 2 else 1 / params["dt"]
            sl = slice(i * 3, i * 3 + 3)
            base = np.linalg.norm((nominal[mask, sl] - y[mask, sl]) * scale, axis=1)
            fitted = np.linalg.norm((pred[mask, sl] - y[mask, sl]) * scale, axis=1)
            result[name] = dict(
                nominal_rms=float(np.sqrt(np.mean(base**2))),
                fitted_rms=float(np.sqrt(np.mean(fitted**2))),
                fitted_worse_rows=int(np.sum(fitted > base)),
                rows=int(np.sum(mask)),
            )
        return result

    records = fit["contrast_sources"]
    assert len(records) == 135
    batches = dict(native=summary(row_kind == 0))
    anchors = []
    for dataset in ("086", "097", "106"):
        ids = [r["equation_group"] for r in records if r["dataset"] == dataset]
        mask = (row_kind == 1) & np.isin(groups, ids)
        assert np.sum(mask) == (9 if dataset == "086" else 63)
        batches[dataset] = summary(mask)
        for group in sorted({r["anchor_group"] for r in records if r["dataset"] == dataset}):
            ids = [r["equation_group"] for r in records if r["dataset"] == dataset and r["anchor_group"] == group]
            mask = (row_kind == 1) & np.isin(groups, ids)
            assert np.sum(mask) == 3
            anchors.append(dict(dataset=dataset, anchor_group=group, metrics=summary(mask)))
    e.recheck()
    output = Path("outputs/wrench_training_residual_audit_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            batches=batches,
            anchors=anchors,
            training_diagnostic_only=True,
            new_fits=0,
            physics_steps=0,
            source_sha256=e.hashes,
        ),
    )
    for key, values in batches.items():
        print(key, {name: (round(v["fitted_rms"], 7), round(v["nominal_rms"], 7)) for name, v in values.items()})
    print(
        "anchors with angular training error above nominal",
        [
            (r["dataset"], r["anchor_group"])
            for r in anchors
            if r["metrics"]["angular_velocity"]["fitted_rms"] > r["metrics"]["angular_velocity"]["nominal_rms"]
        ],
    )


if __name__ == "__main__":
    main()
