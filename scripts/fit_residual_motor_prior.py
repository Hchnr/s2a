"""073 fixed residual prior comparison using independently verified072 rows."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.state_conditioned_gripper_inverse import design_features, reachable_command

BASE = Path("outputs/state_conditioned_gripper_inverse_20261025")
ROOT = Path("outputs/residual_motor_prior_20261025")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths = [
        BASE / name for name in ("fit_result.json", "fit_cache.npz", "parameters.json", "independent_verified.json")
    ]
    paths += [
        Path(__file__),
        Path("src/s2a/state_conditioned_gripper_inverse.py"),
        Path("plans/v0.0.1/073_residual_motor_prior_20261025.md"),
    ]
    hashes = {str(p): digest(p) for p in paths}
    result = json.loads(paths[0].read_text())
    verification = json.loads((BASE / "independent_verified.json").read_text())
    assert verification["numerically_verified"] is True
    assert digest(BASE / "fit_cache.npz") == result["cache_sha256"]
    assert digest(BASE / "parameters.json") == result["parameters_sha256"]
    parameters = json.loads((BASE / "parameters.json").read_text())
    with np.load(BASE / "fit_cache.npz", allow_pickle=False) as data:
        x = design_features(data["raw"], data["available"], parameters["mean"], parameters["scale"])
        y, motor, w = data["target"], data["current_motor"], data["weights"]
    penalty = np.eye(40) * 0.001
    penalty[-1, -1] = 0
    delta = y - motor
    coef = np.linalg.solve(x.T @ (w[:, None] * x) + penalty, x.T @ (w * delta))
    prediction = np.clip(motor + x @ coef, -1, 1)
    selected = np.array(
        [reachable_command(m, p)["selected_motor_target"] for m, p in zip(motor, prediction, strict=True)]
    )
    metrics = []
    for d in result["trajectories"]:
        sl = slice(d["start"], d["stop"])
        metrics.append(
            dict(
                ref=d["ref"],
                task=d["task"],
                split=d["split"],
                predicted_motor_rmse=float(np.sqrt(np.mean((prediction[sl] - y[sl]) ** 2))),
                reachable_motor_rmse=float(np.sqrt(np.mean((selected[sl] - y[sl]) ** 2))),
                hold_motor_rmse=float(np.sqrt(np.mean((motor[sl] - y[sl]) ** 2))),
            )
        )
    held = [m for m in metrics if m["split"] == "holdout"]
    eligible = all(
        m["predicted_motor_rmse"] <= m["hold_motor_rmse"] and m["reachable_motor_rmse"] <= m["hold_motor_rmse"]
        for m in held
    )
    assert all(digest(p) == sha for p, sha in hashes.items())
    ROOT.mkdir(exist_ok=False)
    params = dict(
        parameters, coefficients=coef.tolist(), target="next_motor_minus_current_motor", baseline="current_motor"
    )
    (ROOT / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    output = dict(
        complete=True,
        offline_performance_eligible=eligible,
        physical_qualification=False,
        input_sha256=hashes,
        parameters_sha256=digest(ROOT / "parameters.json"),
        metrics=metrics,
        new_fits=1,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    (ROOT / "fit_result.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(eligible=eligible, holdout=held), indent=2))


if __name__ == "__main__":
    main()
