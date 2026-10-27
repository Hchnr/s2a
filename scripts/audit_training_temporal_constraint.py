"""120 training-only oracle temporal constraint feasibility, never online input."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once
from scipy.spatial.transform import Rotation


def error(a, b):
    relative = Rotation.from_rotvec(a) * Rotation.from_rotvec(b).inv()
    return float(np.sqrt(np.mean(relative.magnitude() ** 2)))


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/120_training_temporal_constraint_20261025.md")
    root = Path("outputs/condition_span_forward_20261025")
    closure = e.json(root / "closure_verified.json")
    assert closure["complete"] and closure["numerically_verified"] and closure["owner_normal_wait_verified"]
    config_path = Path("configs/condition_span_forward_20261025.json")
    e.check(config_path, closure["source_sha256"][str(config_path.resolve())])
    config = e.json(config_path)
    ref = config["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entries = [r for r in e.json(ref["path"])["trajectories"] if r["split"] == "fit"]
    assert len(entries) == 24
    fit_path = root / "fit_result.json"
    e.check(fit_path, closure["source_sha256"][str(fit_path.resolve())])
    fit = e.json(fit_path)
    ref = fit["cache_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        target, base, x = a["target"].copy(), a["nominal"].copy(), a["features"].copy()
    ref = fit["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    coefficients = np.asarray(e.json(ref["path"])["coefficients"])
    predicted = base + x @ coefficients
    offset = 0
    rows = []
    for entry in entries:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as a:
            actual = a["y_rotation_increment"].copy()
            oracle = 0.025 * (a["x_site_twist"][:, 3:] + a["y_twist"][:, 3:])
            n = len(actual)
        np.testing.assert_allclose(actual, target[offset : offset + n, 3:6], rtol=0, atol=1e-12)
        rows.append(
            dict(
                index=entry["index"],
                rows=n,
                oracle_trapezoid_rotation_rmse=error(oracle, actual),
                fitted_rotation_rmse=error(predicted[offset : offset + n, 3:6], actual),
                nominal_rotation_rmse=error(base[offset : offset + n, 3:6], actual),
            )
        )
        offset += n
    assert offset == 4865
    e.recheck()
    output = Path("outputs/training_temporal_constraint_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            diagnostic_only=True,
            oracle_uses_next_label=True,
            fit_trajectories=24,
            natural_rows=4865,
            trajectories=rows,
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    print(
        "oracle <= nominal", sum(r["oracle_trapezoid_rotation_rmse"] <= r["nominal_rotation_rmse"] for r in rows), "/24"
    )
    print(
        "oracle <= fitted", sum(r["oracle_trapezoid_rotation_rmse"] <= r["fitted_rotation_rmse"] for r in rows), "/24"
    )


if __name__ == "__main__":
    main()
