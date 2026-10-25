"""One preregistered072 FP64 ridge fit on fixed native training trajectories."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.state_conditioned_gripper_inverse import design_features, raw_features, reachable_command

ROOT = Path("outputs/state_conditioned_gripper_inverse_20261025")
ROSTER = Path("outputs/gripper_forward_identification_20261025/fit_result.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    roster_sha = digest(ROSTER)
    roster = json.loads(ROSTER.read_text())["trajectories"]
    raw, masks, target, motor, groups, datasets = [], [], [], [], [], []
    for index, entry in enumerate(roster):
        ref = entry["ref"]
        assert digest(ref["path"]) == ref["sha256"]
        start = len(raw)
        with np.load(ref["path"], allow_pickle=False) as data:
            q = data["opening"]
            offsets = data["gripper_current_action_offsets"]
            flat = data["gripper_current_action_flat"]
            motors = []
            for k in range(len(q)):
                cache = flat[offsets[k] : offsets[k + 1]]
                assert cache.shape in ((1,), (2,))
                if len(cache) == 2:
                    assert abs(cache.sum()) < 1e-10
                else:
                    assert k == 0 and cache[0] == 0
                motors.append(float(cache[0]))
            for k in range(len(q) - 1):
                x, mask = raw_features(q[k], motors[k], 0.0 if k == 0 else q[k] - q[k - 1], q[k + 1 : k + 41])
                raw.append(x)
                masks.append(mask)
                motor.append(motors[k])
                target.append(motors[k + 1])
                groups.append(index)
        datasets.append(dict(**entry, start=start, stop=len(raw)))
    x, mask, y, motor, groups = (
        np.asarray(raw),
        np.asarray(masks),
        np.asarray(target),
        np.asarray(motor),
        np.asarray(groups),
    )
    weights = np.zeros(len(x))
    for d in datasets:
        if d["split"] == "fit":
            weights[d["start"] : d["stop"]] = 1 / (24 * (d["stop"] - d["start"]))
    assert sum(d["split"] == "fit" for d in datasets) == 24 and sum(d["split"] == "holdout" for d in datasets) == 6
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
    mean = np.sum(weights[:, None] * x, axis=0)
    scale = np.maximum(np.sqrt(np.sum(weights[:, None] * (x - mean) ** 2, axis=0)), 1e-8)
    features = design_features(x, mask, mean, scale)
    penalty = np.eye(features.shape[1]) * 0.001
    penalty[-1, -1] = 0
    gram = features.T @ (weights[:, None] * features) + penalty
    rhs = features.T @ (weights * y)
    coefficients = np.linalg.solve(gram, rhs)
    predictions = np.clip(features @ coefficients, -1, 1)
    selected = np.array(
        [reachable_command(m, p)["selected_motor_target"] for m, p in zip(motor, predictions, strict=True)]
    )
    metrics = []
    for d in datasets:
        sl = slice(d["start"], d["stop"])
        metrics.append(
            dict(
                ref=d["ref"],
                split=d["split"],
                task=d["task"],
                rows=d["stop"] - d["start"],
                predicted_motor_rmse=float(np.sqrt(np.mean((predictions[sl] - y[sl]) ** 2))),
                reachable_motor_rmse=float(np.sqrt(np.mean((selected[sl] - y[sl]) ** 2))),
                hold_motor_rmse=float(np.sqrt(np.mean((motor[sl] - y[sl]) ** 2))),
            )
        )
    held = [m for m in metrics if m["split"] == "holdout"]
    eligible = all(
        m["predicted_motor_rmse"] <= m["hold_motor_rmse"] and m["reachable_motor_rmse"] <= m["hold_motor_rmse"]
        for m in held
    )
    assert roster_sha == digest(ROSTER) and all(digest(d["ref"]["path"]) == d["ref"]["sha256"] for d in datasets)
    ROOT.mkdir(exist_ok=False)
    np.savez(
        ROOT / "fit_cache.npz", raw=x, available=mask, target=y, current_motor=motor, groups=groups, weights=weights
    )
    parameters = dict(
        seed=42,
        ridge=0.001,
        mean=mean.tolist(),
        scale=scale.tolist(),
        coefficients=coefficients.tolist(),
        feature_dimension=40,
    )
    (ROOT / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    result = dict(
        complete=True,
        offline_performance_eligible=eligible,
        physical_qualification=False,
        roster_ref=dict(path=str(ROSTER.resolve()), sha256=roster_sha),
        trajectories=datasets,
        metrics=metrics,
        parameters_sha256=digest(ROOT / "parameters.json"),
        cache_sha256=digest(ROOT / "fit_cache.npz"),
        source_sha256={p: digest(p) for p in (__file__, "src/s2a/state_conditioned_gripper_inverse.py")},
        new_fits=1,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    (ROOT / "fit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("offline eligible", eligible, "rows", len(x))
    for m in held:
        print(
            Path(m["ref"]["path"]).parent.name,
            m["predicted_motor_rmse"],
            m["reachable_motor_rmse"],
            m["hold_motor_rmse"],
        )


if __name__ == "__main__":
    main()
