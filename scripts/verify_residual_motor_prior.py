"""Independent augmented least squares and scalar reachable-motor audit for073."""

import json
from pathlib import Path

import numpy as np
from fit_residual_motor_prior import digest


def main():
    root = Path("outputs/residual_motor_prior_20261025")
    base = Path("outputs/state_conditioned_gripper_inverse_20261025")
    result = json.loads((root / "fit_result.json").read_text())
    refs = dict(result["input_sha256"])
    refs[str(root / "fit_result.json")] = digest(root / "fit_result.json")
    refs[str(root / "parameters.json")] = result["parameters_sha256"]
    refs[__file__] = digest(__file__)
    assert all(digest(p) == sha for p, sha in refs.items())
    params = json.loads((root / "parameters.json").read_text())
    roster = json.loads((base / "fit_result.json").read_text())["trajectories"]
    with np.load(base / "fit_cache.npz", allow_pickle=False) as data:
        raw, mask, y, motor, w = [data[k] for k in ("raw", "available", "target", "current_motor", "weights")]
    mean = (w[:, None] * raw).sum(axis=0)
    scale = np.maximum(np.sqrt((w[:, None] * (raw - mean) ** 2).sum(axis=0)), 1e-8)
    np.testing.assert_array_equal(mean, params["mean"])
    np.testing.assert_array_equal(scale, params["scale"])
    z = (raw - mean) / scale
    columns = [z[:, i] for i in range(7)]
    columns += [z[:, i] * z[:, j] for i in range(7) for j in range(i, 7)]
    columns += [mask[:, i].astype(float) for i in range(4)] + [np.ones(len(y))]
    x = np.column_stack(columns)
    penalty = np.eye(40) * np.sqrt(0.001)
    penalty[-1, -1] = 0
    a = np.vstack([np.sqrt(w)[:, None] * x, penalty])
    b = np.concatenate([np.sqrt(w) * (y - motor), np.zeros(40)])
    coef = np.linalg.lstsq(a, b, rcond=None)[0]
    diff = float(np.max(np.abs(coef - params["coefficients"])))
    assert diff < 1e-9
    prediction = np.clip(motor + x @ np.asarray(params["coefficients"]), -1, 1)
    reachable = np.clip(motor[:, None] + np.array([0, 0.25, -0.25]), -1, 1)
    selected = reachable[np.arange(len(y)), np.argmin((reachable - prediction[:, None]) ** 2, axis=1)]
    eligible = True
    for d, metrics in zip(roster, result["metrics"], strict=True):
        assert d["ref"] == metrics["ref"] and d["split"] == metrics["split"]
        sl = slice(d["start"], d["stop"])
        errors = [float(np.sqrt(np.mean((v[sl] - y[sl]) ** 2))) for v in (prediction, selected, motor)]
        np.testing.assert_allclose(
            errors,
            [metrics[k] for k in ("predicted_motor_rmse", "reachable_motor_rmse", "hold_motor_rmse")],
            rtol=0,
            atol=1e-12,
        )
        if d["split"] == "holdout":
            eligible &= errors[0] <= errors[2] and errors[1] <= errors[2]
    assert bool(eligible) == result["offline_performance_eligible"]
    assert all(digest(p) == sha for p, sha in refs.items())
    output = dict(
        numerically_verified=True,
        offline_performance_eligible=bool(eligible),
        coefficient_max_abs_difference=diff,
        input_sha256=refs,
        physical_qualification=False,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    with (root / "independent_verified.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in output.items() if k != "input_sha256"}))


if __name__ == "__main__":
    main()
