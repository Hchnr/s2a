"""128 fixed training anchor kernel equation preparation."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.anchor_kernel_forward import condition_features
from s2a.artifacts import sha256


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/128_anchor_kernel_equations_20261025.md")
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
        old_x = cache["features"].copy()
        translation = cache["translation"].copy()
        weights = cache["weights"].copy()
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
    e.read("src/s2a/anchor_kernel_forward.py")
    parameters["anchor_centers"] = np.asarray(states).tolist()
    phi = condition_features(raw, parameters)
    ap = condition_features(states, parameters)
    z = (raw - parameters["state_mean"]) / parameters["state_scale"]
    centers = (np.asarray(states) - parameters["state_mean"]) / parameters["state_scale"]
    blocks = [
        list(range(7)) + list(range(21, 28)),
        list(range(28, 37)),
        list(range(37, 43)),
        list(range(43, 51)),
        list(range(51, 71)),
        list(range(71, 83)),
        list(range(83, 89)),
    ]
    for j in range(66):
        d = sum(np.sum((z[:, b] - centers[j, b]) ** 2, axis=1) / (7 * len(b)) for b in blocks)
        np.testing.assert_allclose(phi[:, 90 + j], (1 + d) ** -2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(phi[:, :84], old_x[:4865, :84], rtol=0, atol=1e-12)
    np.testing.assert_allclose(phi[:, 84:90], old_x[:4865, 212:218], rtol=0, atol=1e-12)
    u = (translation - parameters["translation_mean"]) / parameters["translation_scale"]
    for axis in range(3):
        np.testing.assert_allclose(u[:, axis], old_x[:4865, 282 * (axis + 1)], rtol=0, atol=1e-12)
    natural = np.c_[phi, *(phi * u[:, i, None] for i in range(3))]
    contrast = []
    for j, r in enumerate(records):
        index = next(
            i
            for i, v in enumerate(identities)
            if v["dataset"] == r["dataset"] and v["anchor_group"] == r["anchor_group"]
        )
        du = old_x[4865 + j, [282, 564, 846]]
        goals = []
        for key in ("minus_ref", "plus_ref"):
            ref = r[key]
            e.check(ref["path"], ref["sha256"])
            with np.load(ref["path"], allow_pickle=False) as a:
                goals.append(a["controller_goal_pos"][-1].copy())
        actual = (goals[1] - goals[0]) / r["actual_normalized_delta"] / parameters["translation_scale"]
        np.testing.assert_allclose(du, actual, rtol=1e-8, atol=1e-8)
        contrast.append(np.r_[np.zeros(156), *(ap[index] * d for d in du)])
    x = np.vstack([natural, contrast])
    assert x.shape == (5063, 624)
    parameters.pop("coefficients")
    e.recheck()
    output = Path("outputs/anchor_kernel_equations_20261025")
    output.mkdir(exist_ok=False)
    path = output / "training.npz"
    with path.open("xb") as stream:
        np.savez(stream, features=x, target=y, nominal=base, weights=weights)
    write_once(output / "parameters.json", parameters)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            rows=5063,
            design_dimension=624,
            training_centers=identities,
            training_ref=dict(path=str(path.resolve()), sha256=sha256(path)),
            parameter_ref=dict(
                path=str((output / "parameters.json").resolve()), sha256=sha256(output / "parameters.json")
            ),
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    print("128 all5063x624 features and66 fit centers verified;fit0")


if __name__ == "__main__":
    main()
