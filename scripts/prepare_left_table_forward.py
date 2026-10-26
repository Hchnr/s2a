"""145 bind qualified144 data to the unchanged116 physical response model."""

from pathlib import Path

import numpy as np
from arm_left_table_independent import equations
from arm_left_table_training import contrast_rows
from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def main():
    e = Evidence()
    e.read(__file__)
    for p in (
        "scripts/arm_left_table_training.py",
        "scripts/arm_left_table_independent.py",
        "plans/v0.0.1/145_left_table_forward_20261025.md",
    ):
        e.read(p)
    original = Path("outputs/condition_span_forward_20261025")
    old = e.json(original / "closure_verified.json")
    assert old["complete"] and old["numerically_verified"] and old["owner_normal_wait_verified"]
    path = Path("outputs/left_table_interventions_20261025/closure_verified.json")
    closed = e.json(path)
    assert (
        closed["complete"]
        and closed["data_qualified"]
        and closed["verified_cases"] == 21
        and closed["physics_steps"] == 1309
        and closed["owner_normal_wait_verified"]
    )
    for closure in (old, closed):
        for p, h in closure["source_sha256"].items():
            e.check(p, h)
    config = e.json("configs/condition_span_forward_20261025.json")
    config["left_table_closure_ref"] = dict(path=str(path.resolve()), sha256=sha256(path))
    p = original / "fit_result.json"
    e.check(p, old["source_sha256"][str(p.resolve())])
    fit = e.json(p)
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    parameters = e.json(fit["parameter_ref"]["path"])
    with np.load(fit["cache_ref"]["path"], allow_pickle=False) as a:
        x, y, b, w = (a[k].copy() for k in ("features", "target", "nominal", "weights"))
    cx, cy, cb, records = contrast_rows(config, e, parameters)
    ix, iy, ib, ids = equations(config, e, parameters)
    for a, other, tol in ((cx, ix, 1e-8), (cy, iy, 1e-10), (cb, ib, 1e-10)):
        np.testing.assert_allclose(a, other, rtol=0, atol=tol)
    assert ids == [(r["anchor_group"], r["axis"], r["minus_case"], r["plus_case"]) for r in records]
    for r in records:
        r.update(dataset="144", equation_group=4000 + 3 * r["anchor_group"] + r["axis"])
    all_records = fit["contrast_sources"] + records
    assert len(all_records) == 207 and len({r["equation_group"] for r in all_records}) == 207
    x = np.vstack([x, cx])
    y = np.vstack([y, cy])
    b = np.vstack([b, cb])
    w = np.r_[w[:4865], np.full(207, 0.5 / 207)]
    assert x.shape == (5072, 1128) and y.shape == b.shape == (5072, 12)
    np.testing.assert_allclose([w[:4865].sum(), w[4865:].sum()], [0.5, 0.5], rtol=0, atol=1e-12)
    e.recheck()
    out = Path("outputs/left_table_forward_equations_20261025")
    out.mkdir(exist_ok=False)
    path = out / "training.npz"
    with path.open("xb") as stream:
        np.savez(stream, features=x, target=y, nominal=b, weights=w)
    parameters.pop("coefficients")
    write_once(out / "parameters.json", parameters)
    config.update(
        output_dir="outputs/left_table_forward_20261025", contrast_equations=207, planned_total_equations=5072
    )
    write_once("configs/left_table_forward_20261025.json", config)
    write_once(
        out / "result.json",
        dict(
            complete=True,
            rows=5072,
            design_dimension=1128,
            contrast_equations=207,
            contrast_sources=all_records,
            training_ref=dict(path=str(path.resolve()), sha256=sha256(path)),
            parameter_ref=dict(path=str((out / "parameters.json").resolve()), sha256=sha256(out / "parameters.json")),
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    print("145 original5063 qualified equations preserved;new9 independent parity;5072 total;fit0")


if __name__ == "__main__":
    main()
