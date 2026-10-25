"""Independent072 raw-row reconstruction and augmented least-squares proof."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/state_conditioned_gripper_inverse_20261025")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    fit_path = ROOT / "fit_result.json"
    parameter_path = ROOT / "parameters.json"
    cache_path = ROOT / "fit_cache.npz"
    fit_sha = digest(fit_path)
    fit = json.loads(fit_path.read_text())
    parameters = json.loads(parameter_path.read_text())
    assert digest(parameter_path) == fit["parameters_sha256"] and digest(cache_path) == fit["cache_sha256"]
    assert parameters["ridge"] == 0.001 and parameters["feature_dimension"] == 40
    for path, sha in fit["source_sha256"].items():
        assert digest(path) == sha
    roster = fit["roster_ref"]
    assert digest(roster["path"]) == roster["sha256"]
    source_roster = json.loads(Path(roster["path"]).read_text())["trajectories"]
    assert [(d["ref"], d["split"]) for d in source_roster] == [(d["ref"], d["split"]) for d in fit["trajectories"]]
    raw, masks, target, motors, groups, weights = [], [], [], [], [], []
    for index, entry in enumerate(fit["trajectories"]):
        ref = entry["ref"]
        assert digest(ref["path"]) == ref["sha256"]
        assert entry["start"] == len(raw)
        with np.load(ref["path"], allow_pickle=False) as data:
            q = data["opening"]
            offset = data["gripper_current_action_offsets"]
            flat = data["gripper_current_action_flat"]
            for k in range(len(q) - 1):
                current = flat[offset[k] : offset[k + 1]]
                following = flat[offset[k + 1] : offset[k + 2]]
                assert current.shape in ((1,), (2,)) and following.shape == (2,)
                values = [float(q[k]), float(current[0]), float(q[k] - q[k - 1]) if k else 0.0]
                available = []
                for ahead in (1, 2, 4, 8):
                    endpoint = min(k + ahead, len(q) - 1)
                    values.append(float(q[endpoint] - q[k]))
                    available.append(k + ahead < len(q))
                raw.append(values)
                masks.append(available)
                target.append(float(following[0]))
                motors.append(float(current[0]))
                groups.append(index)
                weights.append(1 / (24 * (len(q) - 1)) if entry["split"] == "fit" else 0.0)
        assert entry["stop"] == len(raw)
    raw, masks, target, motors, groups, weights = map(np.asarray, (raw, masks, target, motors, groups, weights))
    with np.load(cache_path, allow_pickle=False) as cache:
        for name, value in dict(
            raw=raw, available=masks, target=target, current_motor=motors, groups=groups, weights=weights
        ).items():
            np.testing.assert_array_equal(cache[name], value)
    mean = weights @ raw
    scale = np.maximum(np.sqrt(weights @ ((raw - mean) ** 2)), 1e-8)
    np.testing.assert_allclose(parameters["mean"], mean, atol=1e-12, rtol=0)
    np.testing.assert_allclose(parameters["scale"], scale, atol=1e-12, rtol=0)
    z = (raw - mean) / scale
    features = np.column_stack(
        [
            *[z[:, i] for i in range(7)],
            *[z[:, i] * z[:, j] for i in range(7) for j in range(i, 7)],
            *[masks[:, i].astype(float) for i in range(4)],
            np.ones(len(raw)),
        ]
    )
    ridge = np.eye(40) * np.sqrt(0.001)
    ridge[-1, -1] = 0
    augmented = np.vstack([features * np.sqrt(weights)[:, None], ridge])
    rhs = np.r_[target * np.sqrt(weights), np.zeros(40)]
    coefficients = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    coefficient_error = float(np.max(np.abs(coefficients - np.asarray(parameters["coefficients"]))))
    assert coefficient_error < 1e-9
    gradient = augmented.T @ (augmented @ coefficients - rhs)
    assert np.max(np.abs(gradient)) < 1e-10
    prediction = np.clip(features @ coefficients, -1, 1)
    reachable = np.clip(motors[:, None] + np.array([0.0, 0.25, -0.25]), -1, 1)
    selected = reachable[np.arange(len(raw)), np.argmin((reachable - prediction[:, None]) ** 2, axis=1)]
    metrics = []
    for entry, reported in zip(fit["trajectories"], fit["metrics"], strict=True):
        sl = slice(entry["start"], entry["stop"])
        computed = dict(
            predicted_motor_rmse=float(np.sqrt(np.mean((prediction[sl] - target[sl]) ** 2))),
            reachable_motor_rmse=float(np.sqrt(np.mean((selected[sl] - target[sl]) ** 2))),
            hold_motor_rmse=float(np.sqrt(np.mean((motors[sl] - target[sl]) ** 2))),
        )
        for key, value in computed.items():
            np.testing.assert_allclose(value, reported[key], atol=1e-10, rtol=0)
        metrics.append(dict(ref=entry["ref"], split=entry["split"], **computed))
    eligible = all(
        d["predicted_motor_rmse"] <= d["hold_motor_rmse"] and d["reachable_motor_rmse"] <= d["hold_motor_rmse"]
        for d in metrics
        if d["split"] == "holdout"
    )
    assert eligible == fit["offline_performance_eligible"]
    assert fit_sha == digest(fit_path) and all(
        digest(d["ref"]["path"]) == d["ref"]["sha256"] for d in fit["trajectories"]
    )
    output = dict(
        complete=True,
        numerically_verified=True,
        offline_performance_eligible=eligible,
        coefficient_max_abs_difference=coefficient_error,
        gradient_max_abs=float(np.max(np.abs(gradient))),
        verified_rows=len(raw),
        fit_sha256=fit_sha,
        script_sha256=digest(__file__),
        metrics=metrics,
        scope="Independent reconstruction/solve of same candidate; not a new fit or physical qualification",
        new_candidate_fits=0,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    with (ROOT / "independent_verified.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: output[k]
                for k in (
                    "numerically_verified",
                    "offline_performance_eligible",
                    "verified_rows",
                    "coefficient_max_abs_difference",
                    "gradient_max_abs",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
