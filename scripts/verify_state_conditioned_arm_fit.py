"""088 independent fit-row assembly and augmented SVD verification."""

import json
from pathlib import Path

import numpy as np
from arm_forward_independent import conditions, feature_matrix, nominal_prediction
from audit_wam_position import Evidence

ROOT = Path("outputs/state_conditioned_arm_forward_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    e.read("scripts/arm_forward_independent.py")
    config = e.json("configs/state_conditioned_arm_forward_20261025.json")
    fit = e.json(ROOT / "fit_result.json")
    assert fit["complete"] and fit["new_fits"] == 1 and fit["rows"] == 4865
    assert fit["holdout_rows_used"] == fit["probe_rows_used"] == 0
    for key in ("parameter_ref", "cache_ref"):
        ref = fit[key]
        e.check(ref["path"], ref["sha256"])
    for path, signature in fit["source_sha256"].items():
        e.check(path, signature)
    params = e.json(fit["parameter_ref"]["path"])
    assert params["ridge"] == config["ridge"] == 0.001
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
    assert x.shape == (4865, 288) and y.shape == (4865, 12)
    for values, mean_key, scale_key in [
        (raw, "state_mean", "state_scale"),
        (actions, "translation_mean", "translation_scale"),
    ]:
        mu = np.average(values, axis=0, weights=w)
        scale = np.sqrt(np.average((values - mu) ** 2, axis=0, weights=w))
        scale[scale < 1e-8] = 1
        np.testing.assert_allclose(params[mean_key], mu, rtol=0, atol=1e-12)
        np.testing.assert_allclose(params[scale_key], scale, rtol=0, atol=1e-12)
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
        ]:
            np.testing.assert_allclose(cache[key], value, rtol=1e-10, atol=1e-10, err_msg=key)
    penalty = np.eye(288)
    penalty[0, 0] = 0
    matrix = np.vstack([x * np.sqrt(w)[:, None], np.sqrt(0.001) * penalty])
    rhs = np.vstack([residual * np.sqrt(w)[:, None], np.zeros((288, 12))])
    recomputed, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    assert rank == 288
    produced = np.asarray(params["coefficients"])
    np.testing.assert_allclose(produced, recomputed, rtol=1e-8, atol=1e-8)
    gradient = x.T @ (w[:, None] * (x @ produced - residual)) + 0.001 * penalty @ produced
    max_gradient = float(np.max(np.abs(gradient)))
    assert max_gradient < 1e-9
    e.recheck()
    result = dict(
        complete=True,
        numerically_verified=True,
        rows=4865,
        fit_trajectories=24,
        rank=int(rank),
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
    print("088 independent4865rows/features/weights/augmentedSVD passed")


if __name__ == "__main__":
    main()
