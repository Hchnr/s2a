"""116 independent native/contrast equation and augmented SVD verification."""

import json
from pathlib import Path

import numpy as np
from arm_condition_span_combined_independent import equations
from arm_joint_wrench_independent import augment, conditions, feature_matrix, nominal_prediction
from arm_wrench_data_independent import attach_current_wrench
from audit_wam_position import Evidence

ROOT = Path("outputs/condition_span_forward_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    e.read("scripts/arm_wrench_data.py")
    e.read("scripts/arm_wrench_data_independent.py")
    e.read("scripts/arm_joint_wrench_independent.py")
    e.read("src/s2a/joint_wrench_condition_forward.py")
    e.read("scripts/arm_nonlinear_independent.py")
    e.read("scripts/arm_joint_wrench_independent_equations.py")
    for module in (
        "scripts/arm_condition_span_combined_training.py",
        "scripts/arm_condition_span_combined_independent.py",
        "scripts/arm_joint_wrench_coverage_training.py",
        "scripts/arm_joint_wrench_coverage_independent.py",
    ):
        e.read(module)
    for module in (
        "scripts/arm_condition_span_training.py",
        "scripts/arm_condition_span_independent.py",
        "scripts/arm_condition_span_combined_training.py",
        "scripts/arm_condition_span_combined_independent.py",
    ):
        e.read(module)
    config = e.json("configs/condition_span_forward_20261025.json")
    fit = e.json(ROOT / "fit_result.json")
    assert fit["complete"] and fit["new_fits"] == 1 and fit["rows"] == 5063
    assert fit["holdout_rows_used"] == fit["held_anchor_groups_used"] == 0
    assert fit["probe_rows_used"] == 396 and fit["native_rows"] == 4865 and fit["contrast_equations"] == 198
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    for path, signature in fit["source_sha256"].items():
        e.check(path, signature)
    params = e.json(fit["parameter_ref"]["path"])
    np.testing.assert_array_equal(params["frequencies"], config["frequencies"])
    np.testing.assert_array_equal(params["joint_frequencies"], config["joint_frequencies"])
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
        data = attach_current_wrench(data, entry, config, e)
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
    assert x.shape == (4865, 1128) and y.shape == (4865, 12)
    for values, mean_key, scale_key in [
        (raw, "state_mean", "state_scale"),
        (actions, "translation_mean", "translation_scale"),
    ]:
        mu = np.average(values, axis=0, weights=w)
        scale = np.sqrt(np.average((values - mu) ** 2, axis=0, weights=w))
        scale[scale < 1e-8] = 1
        if mean_key == "state_mean":
            np.testing.assert_allclose(params[mean_key][:83], mu[:83], rtol=0, atol=1e-12)
            np.testing.assert_allclose(params[scale_key][:83], scale[:83], rtol=0, atol=1e-12)
            np.testing.assert_allclose(params[mean_key][83:], mu[83:], rtol=0, atol=1e-10)
            np.testing.assert_allclose(params[scale_key][83:], scale[83:], rtol=0, atol=1e-10)
        else:
            np.testing.assert_allclose(params[mean_key], mu, rtol=0, atol=1e-12)
            np.testing.assert_allclose(params[scale_key], scale, rtol=0, atol=1e-12)
    ref = config["statistics_ref"]
    e.check(ref["path"], ref["sha256"])
    old_stats = e.json(ref["path"])
    for key in ("state_mean", "state_scale", "translation_mean", "translation_scale"):
        if key.startswith("state_"):
            np.testing.assert_array_equal(params[key][:83], old_stats[key])
            np.testing.assert_array_equal(
                params[key][83:], config["wrench_mean" if key == "state_mean" else "wrench_scale"]
            )
        else:
            np.testing.assert_array_equal(params[key], old_stats[key])
    ref = config["original_parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    original_parameters = e.json(ref["path"])
    for key in ("state_mean", "state_scale", "translation_mean", "translation_scale", "frequencies"):
        np.testing.assert_array_equal(params[key], original_parameters[key])
    cx, cy, cn, ids = equations(config, e, params)
    assert ids == [
        (r["dataset"], r["anchor_group"], r["axis"], r["minus_case"], r["plus_case"]) for r in fit["contrast_sources"]
    ]
    x = np.vstack([x, cx])
    y = np.vstack([y, cy])
    nominal = np.vstack([nominal, cn])
    w = np.r_[w * 0.5, np.full(198, 0.5 / 198)]
    groups = np.r_[
        groups,
        [
            {"086": 100, "097": 1000, "106": 2000, "115": 3000}[dataset] + group * 3 + axis
            for dataset, group, axis, _, _ in ids
        ],
    ]
    row_kind = np.r_[np.zeros(4865, dtype=np.int64), np.ones(198, dtype=np.int64)]
    assert x.shape == (5063, 1128) and abs(w[:4865].sum() - 0.5) < 1e-12 and abs(w[4865:].sum() - 0.5) < 1e-12
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
    penalty = np.eye(1128)
    penalty[0, 0] = 0
    matrix = np.vstack([x * np.sqrt(w)[:, None], np.sqrt(0.001) * penalty])
    rhs = np.vstack([residual * np.sqrt(w)[:, None], np.zeros((1128, 12))])
    recomputed, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    assert rank == 1128
    produced = np.asarray(params["coefficients"])
    np.testing.assert_allclose(produced, recomputed, rtol=1e-8, atol=1e-8)
    gradient = x.T @ (w[:, None] * (x @ produced - residual)) + 0.001 * penalty @ produced
    max_gradient = float(np.max(np.abs(gradient)))
    assert max_gradient < 1e-9
    e.recheck()
    result = dict(
        complete=True,
        numerically_verified=True,
        rows=5063,
        native_rows=4865,
        contrast_equations=198,
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
    print("116 independent4865native+198contrast/weights/augmentedSVD passed")


if __name__ == "__main__":
    main()
