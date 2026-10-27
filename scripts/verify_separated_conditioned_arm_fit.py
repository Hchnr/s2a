"""101 independent native/contrast equation and augmented SVD verification."""

import json
from pathlib import Path

import numpy as np
from arm_nonlinear_independent import augment, conditions, feature_matrix, nominal_prediction
from arm_nonlinear_independent_equations import equations
from audit_wam_position import Evidence

ROOT = Path("outputs/separated_conditioned_arm_forward_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    e.read("scripts/arm_nonlinear_independent.py")
    e.read("scripts/arm_nonlinear_independent_equations.py")
    config = e.json("configs/separated_conditioned_arm_forward_20261025.json")
    fit = e.json(ROOT / "fit_result.json")
    assert fit["complete"] and fit["new_fits"] == 2 and fit["rows"] == 4937
    assert fit["holdout_rows_used"] == fit["held_anchor_groups_used"] == 0
    assert fit["probe_rows_used"] == 144 and fit["native_rows"] == 4865 and fit["contrast_equations"] == 72
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    for path, signature in fit["source_sha256"].items():
        e.check(path, signature)
    params = e.json(fit["parameter_ref"]["path"])
    np.testing.assert_array_equal(params["frequencies"], config["frequencies"])
    assert params["objective_mode"] == "separate_intervention_slopes_then_native_offset"
    assert params["ridge"] == config["ridge"] == 0.001
    assert params["training_anchor_groups"] == [0, 3, 5] and params["validation_anchor_groups"] == [1, 2, 4]
    assert params["native_weight"] == params["contrast_weight"] == 1.0
    source = e.json(config["data_result_ref"]["path"])
    entries = [p for p in source["trajectories"] if p["split"] == "fit"]
    assert len(entries) == 24
    assert [p["index"] for p in entries] == [p["index"] for p in fit["fit_sources"]]
    raw = []
    actions = []
    weights = []
    groups = []
    features = []
    labels = []
    base = []
    for entry in entries:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            data = {k: cache[k].copy() for k in cache.files}
        data = augment(data, params)
        n = len(data["x_joint_position"])
        raw.append(conditions(data))
        actions.append(data["action_goal_delta"][:, :3])
        weights.extend([1 / (24 * n)] * n)
        groups.extend([entry["index"]] * n)
        features.append(feature_matrix(data, params))
        base.append(nominal_prediction(data, params))
        labels.append(np.concatenate([data["y_motion"], data["y_rotation_increment"], 0.05 * data["y_twist"]], axis=1))
    raw = np.concatenate(raw)
    actions = np.concatenate(actions)
    w = np.array(weights)
    x = np.concatenate(features)
    y = np.concatenate(labels)
    nominal = np.concatenate(base)
    assert x.shape == (4865, 848) and y.shape == (4865, 12)
    for values, mean_key, scale_key in [
        (raw, "state_mean", "state_scale"),
        (actions, "translation_mean", "translation_scale"),
    ]:
        mu = np.average(values, axis=0, weights=w)
        scale = np.sqrt(np.average((values - mu) ** 2, axis=0, weights=w))
        scale[scale < 1e-8] = 1
        np.testing.assert_allclose(params[mean_key], mu, rtol=0, atol=1e-12)
        np.testing.assert_allclose(params[scale_key], scale, rtol=0, atol=1e-12)
    ref = config["statistics_ref"]
    e.check(ref["path"], ref["sha256"])
    old_stats = e.json(ref["path"])
    for key in ("state_mean", "state_scale", "translation_mean", "translation_scale"):
        np.testing.assert_array_equal(params[key], old_stats[key])
    cx, cy, cn, ids = equations(config, e, params)
    assert ids == [
        (r["dataset"], r["anchor_group"], r["axis"], r["minus_case"], r["plus_case"]) for r in fit["contrast_sources"]
    ]
    x = np.vstack([x, cx])
    y = np.vstack([y, cy])
    nominal = np.vstack([nominal, cn])
    w = np.r_[w, np.full(72, 1.0 / 72)]
    groups = np.r_[
        groups, [(100 if dataset == "086" else 1000) + group * 3 + axis for dataset, group, axis, _, _ in ids]
    ]
    row_kind = np.r_[np.zeros(4865, dtype=np.int64), np.ones(72, dtype=np.int64)]
    assert x.shape == (4937, 848) and abs(w[:4865].sum() - 1.0) < 1e-12 and abs(w[4865:].sum() - 1.0) < 1e-12
    residual = y - nominal
    with np.load(fit["cache_ref"]["path"], allow_pickle=False) as cache:
        for key, value in [
            ("raw", raw),
            ("translation", actions),
            ("features", x),
            ("target", y),
            ("nominal", nominal),
            ("residual_target", residual),
            ("weights", w),
            ("groups", groups),
            ("row_kind", row_kind),
        ]:
            if key in ("groups", "row_kind", "weights"):
                np.testing.assert_array_equal(cache[key], value, err_msg=key)
            else:
                tolerance = 1e-8 if key == "features" else 1e-10
                np.testing.assert_allclose(cache[key], value, rtol=tolerance, atol=tolerance, err_msg=key)
    assert np.max(np.abs(x[4865:, :212])) == 0
    sx, sy, sw = x[4865:, 212:], residual[4865:], w[4865:]
    sm = np.vstack([sx * np.sqrt(sw)[:, None], np.sqrt(0.001) * np.eye(636)])
    sr = np.vstack([sy * np.sqrt(sw)[:, None], np.zeros((636, 12))])
    slopes, _, slope_rank, _ = np.linalg.lstsq(sm, sr, rcond=None)
    assert slope_rank == 636
    nx, nw = x[:4865, :212], w[:4865]
    ny = residual[:4865] - x[:4865, 212:] @ slopes
    penalty = np.eye(212)
    penalty[0, 0] = 0
    nm = np.vstack([nx * np.sqrt(nw)[:, None], np.sqrt(0.001) * penalty])
    nr = np.vstack([ny * np.sqrt(nw)[:, None], np.zeros((212, 12))])
    offsets, _, offset_rank, _ = np.linalg.lstsq(nm, nr, rcond=None)
    assert offset_rank == 212
    recomputed = np.vstack([offsets, slopes])
    produced = np.asarray(params["coefficients"])
    np.testing.assert_allclose(produced, recomputed, rtol=1e-8, atol=1e-8)
    slope_gradient = sx.T @ (sw[:, None] * (sx @ produced[212:] - sy)) + 0.001 * produced[212:]
    conditional_target = residual[:4865] - x[:4865, 212:] @ produced[212:]
    offset_gradient = (
        nx.T @ (nw[:, None] * (nx @ produced[:212] - conditional_target)) + 0.001 * penalty @ produced[:212]
    )
    slope_max = float(np.max(np.abs(slope_gradient)))
    offset_max = float(np.max(np.abs(offset_gradient)))
    max_gradient = max(slope_max, offset_max)
    assert max_gradient < 1e-9
    rank = slope_rank + offset_rank
    e.recheck()
    result = dict(
        complete=True,
        numerically_verified=True,
        rows=4937,
        native_rows=4865,
        contrast_equations=72,
        fit_trajectories=24,
        rank=int(rank),
        history_dimension=12,
        max_coefficient_absolute_difference=float(np.max(np.abs(produced - recomputed))),
        maximum_regularized_gradient=max_gradient,
        slope_gradient_max=slope_max,
        conditional_offset_gradient_max=offset_max,
        source_sha256=e.hashes,
        new_parameter_fits=0,
        independent_numerical_solves=2,
        separate_stage_verification=True,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (ROOT / "independent_fit_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("101 independent4865native+72contrast/weights/augmentedSVD passed")


if __name__ == "__main__":
    main()
