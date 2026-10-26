"""090 independent native/contrast equation and augmented SVD verification."""

import json
from pathlib import Path

import numpy as np
from arm_history_contrast_independent import equations
from arm_history_independent import augment, conditions, feature_matrix, nominal_prediction
from audit_wam_position import Evidence

ROOT = Path("outputs/history_conditioned_arm_forward_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    e.read("scripts/arm_history_independent.py")
    e.read("scripts/arm_history_contrast_independent.py")
    config = e.json("configs/history_conditioned_arm_forward_20261025.json")
    fit = e.json(ROOT / "fit_result.json")
    assert fit["complete"] and fit["new_fits"] == 1 and fit["rows"] == 4874
    assert fit["holdout_rows_used"] == fit["held_anchor_groups_used"] == 0
    assert fit["probe_rows_used"] == 18 and fit["native_rows"] == 4865 and fit["contrast_equations"] == 9
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    for path, signature in fit["source_sha256"].items():
        e.check(path, signature)
    params = e.json(fit["parameter_ref"]["path"])
    assert params["ridge"] == config["ridge"] == 0.001
    assert params["training_anchor_groups"] == [0, 3, 5] and params["validation_anchor_groups"] == [1, 2, 4]
    assert params["native_weight"] == params["contrast_weight"] == 0.5
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
    assert x.shape == (4865, 336) and y.shape == (4865, 12)
    for values, mean_key, scale_key in [
        (raw, "state_mean", "state_scale"),
        (actions, "translation_mean", "translation_scale"),
    ]:
        mu = np.average(values, axis=0, weights=w)
        scale = np.sqrt(np.average((values - mu) ** 2, axis=0, weights=w))
        scale[scale < 1e-8] = 1
        np.testing.assert_allclose(params[mean_key], mu, rtol=0, atol=1e-12)
        np.testing.assert_allclose(params[scale_key], scale, rtol=0, atol=1e-12)
    cx, cy, cn, ids = equations(config, e, params)
    assert ids == [(r["anchor_group"], r["axis"], r["minus_case"], r["plus_case"]) for r in fit["contrast_sources"]]
    x = np.vstack([x, cx])
    y = np.vstack([y, cy])
    nominal = np.vstack([nominal, cn])
    w = np.r_[w * 0.5, np.full(9, 0.5 / 9)]
    groups = np.r_[groups, [100 + group * 3 + axis for group, axis, _, _ in ids]]
    row_kind = np.r_[np.zeros(4865, dtype=np.int64), np.ones(9, dtype=np.int64)]
    assert x.shape == (4874, 336) and abs(w[:4865].sum() - 0.5) < 1e-12 and abs(w[4865:].sum() - 0.5) < 1e-12
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
    penalty = np.eye(336)
    penalty[0, 0] = 0
    matrix = np.vstack([x * np.sqrt(w)[:, None], np.sqrt(0.001) * penalty])
    rhs = np.vstack([residual * np.sqrt(w)[:, None], np.zeros((336, 12))])
    recomputed, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    assert rank == 336
    produced = np.asarray(params["coefficients"])
    np.testing.assert_allclose(produced, recomputed, rtol=1e-8, atol=1e-8)
    gradient = x.T @ (w[:, None] * (x @ produced - residual)) + 0.001 * penalty @ produced
    max_gradient = float(np.max(np.abs(gradient)))
    assert max_gradient < 1e-9
    e.recheck()
    result = dict(
        complete=True,
        numerically_verified=True,
        rows=4874,
        native_rows=4865,
        contrast_equations=9,
        fit_trajectories=24,
        rank=int(rank),
        history_dimension=12,
        max_coefficient_absolute_difference=float(np.max(np.abs(produced - recomputed))),
        maximum_regularized_gradient=max_gradient,
        source_sha256=e.hashes,
        new_parameter_fits=0,
        independent_numerical_solves=1,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (ROOT / "independent_fit_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("090 independent4865native+9contrast/weights/augmentedSVD passed")


if __name__ == "__main__":
    main()
