"""Independent scalar reconstruction and normal-equation check for065."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/gripper_forward_identification_20261025")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result_path = ROOT / "fit_result.json"
    fit_sha = digest(result_path)
    fit = json.loads(result_path.read_text())
    coef = np.asarray(fit["coefficients"])
    records, xfit, yfit, consumed = [], [], [], {}
    for trajectory in fit["trajectories"]:
        ref = trajectory["ref"]
        path = Path(ref["path"])
        assert digest(path) == ref["sha256"]
        consumed[str(path)] = ref["sha256"]
        x, y, active, steps = [], [], [], []
        with np.load(path, allow_pickle=False) as data:
            opening = data["opening"]
            offsets = data["gripper_current_action_offsets"]
            cache = data["gripper_current_action_flat"]
            for k in range(1, len(opening) - 1):
                motors = []
                for index in (k, k + 1):
                    pair = cache[offsets[index] : offsets[index + 1]]
                    assert pair.shape == (2,)
                    left, right = 0.02 + 0.02 * pair[0], -0.02 + 0.02 * pair[1]
                    motors.append(float(np.clip((left - right - 0.001) / 0.079, 0, 1)))
                x.append([motors[1] - opening[k], opening[k] - opening[k - 1]])
                y.append(opening[k + 1] - opening[k])
                loaded = motors[1] >= motors[0] - 1e-12 and motors[1] > opening[k] + 0.02
                active.append(loaded)
                if loaded:
                    steps.append(k)
        assert steps == trajectory["loaded_steps"]
        x, y, active = np.asarray(x), np.asarray(y), np.asarray(active)
        if trajectory["split"] == "fit":
            xfit.extend(x[active])
            yfit.extend(y[active])
        records.append(
            dict(
                task=trajectory["task"],
                split=trajectory["split"],
                ref=ref,
                opening_rows=int(active.sum()),
                opening_rmse=float(np.sqrt(np.mean((x[active] @ coef - y[active]) ** 2))),
                opening_hold_rmse=float(np.sqrt(np.mean(y[active] ** 2))),
                all_rmse=float(np.sqrt(np.mean((x @ coef - y) ** 2))),
            )
        )
    xfit, yfit = np.asarray(xfit), np.asarray(yfit)
    independent = np.linalg.lstsq(xfit, yfit, rcond=None)[0]
    np.testing.assert_allclose(coef, independent, atol=1e-10, rtol=0)
    gradient = xfit.T @ (xfit @ coef - yfit)
    assert np.max(np.abs(gradient)) < 1e-9 and np.all((coef > 0) & (coef < 1))
    heldout = [r for r in records if r["split"] == "holdout"]
    assert len(heldout) == 6
    assert digest(result_path) == fit_sha and all(digest(p) == sha for p, sha in consumed.items())
    result = dict(
        passed=True,
        fit_sha256=fit_sha,
        independent_coefficients=independent.tolist(),
        normal_gradient_max=float(np.max(np.abs(gradient))),
        trajectories=records,
        holdout_opening_improved=sum(r["opening_rmse"] < r["opening_hold_rmse"] for r in heldout),
        holdout_trials=6,
        input_sha256=consumed,
        script_sha256=digest(__file__),
        new_fits=0,
        note="Independent unconstrained solve certifies interior bounded fit; no new candidate fit",
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    with (ROOT / "independent_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("holdout improved", result["holdout_opening_improved"], "/6", "gradient", result["normal_gradient_max"])
    for r in heldout:
        print(
            r["task"], Path(r["ref"]["path"]).parent.name, r["opening_rows"], r["opening_rmse"], r["opening_hold_rmse"]
        )


if __name__ == "__main__":
    main()
