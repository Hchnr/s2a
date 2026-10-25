"""Independent075 raw-row, augmented least-squares and scalar rollout verification."""

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def advance(m, u):
    increment = -0.01 if u > 0 else (0.01 if u < 0 else 0.0)
    for _ in range(25):
        m = max(-1.0, min(1.0, m + increment))
    return m


def features(values, mean, scale):
    z = [(v - m) / s for v, m, s in zip(values, mean, scale, strict=True)]
    return np.array(z + [z[i] * z[j] for i in range(4) for j in range(i, 4)] + [1.0])


def main():
    root = Path("outputs/all_phase_gripper_forward_20261025")
    fit = json.loads((root / "fit_result.json").read_text())
    params = json.loads((root / "parameters.json").read_text())
    refs = dict(fit["source_sha256"])
    refs.update(
        {
            str(root / "fit_result.json"): digest(root / "fit_result.json"),
            str(root / "parameters.json"): fit["parameters_sha256"],
            str(root / "fit_cache.npz"): fit["cache_sha256"],
            __file__: digest(__file__),
        }
    )
    assert all(digest(p) == s for p, s in refs.items())
    roster = json.loads(Path("outputs/gripper_forward_identification_20261025/fit_result.json").read_text())[
        "trajectories"
    ]
    raw, y, w, trajectories = [], [], [], []
    for entry in roster:
        with np.load(entry["ref"]["path"], allow_pickle=False) as data:
            q = data["opening"].copy()
            u = data["sent_commands"][:, -1].copy()
            offsets = data["gripper_current_action_offsets"]
            flat = data["gripper_current_action_flat"]
            m = [float(flat[offsets[k]]) for k in range(len(q))]
        for k in range(len(u)):
            nextmotor = advance(m[k], u[k])
            assert abs(nextmotor - m[k + 1]) < 1e-12
            raw.append([q[k], m[k], nextmotor, 0.0 if k == 0 else q[k] - q[k - 1]])
            y.append(q[k + 1] - q[k])
            w.append(1 / (24 * len(u)) if entry["split"] == "fit" else 0.0)
        trajectories.append((entry, q, u, m))
    raw, y, w = np.array(raw), np.array(y), np.array(w)
    with np.load(root / "fit_cache.npz", allow_pickle=False) as cached:
        for value, key in [(raw, "raw"), (y, "target_delta_q"), (w, "weights")]:
            np.testing.assert_allclose(value, cached[key], rtol=0, atol=1e-12)
    mean = (w[:, None] * raw).sum(axis=0)
    scale = np.maximum(np.sqrt((w[:, None] * (raw - mean) ** 2).sum(axis=0)), 1e-8)
    np.testing.assert_allclose(mean, params["mean"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(scale, params["scale"], rtol=0, atol=1e-12)
    x = np.array([features(row, mean, scale) for row in raw])
    penalty = np.eye(15) * np.sqrt(0.001)
    penalty[-1, -1] = 0
    a = np.vstack([np.sqrt(w)[:, None] * x, penalty])
    b = np.r_[np.sqrt(w) * y, np.zeros(15)]
    independent = np.linalg.lstsq(a, b, rcond=None)[0]
    coef = np.array(params["coefficients"])
    difference = float(np.max(np.abs(coef - independent)))
    assert difference < 1e-9
    prediction = np.clip(raw[:, 0] + x @ coef, 0, 1)
    start = 0
    eligible = True
    for (entry, q, u, m), reported in zip(trajectories, fit["metrics"], strict=True):
        assert entry["ref"] == reported["ref"]
        errors, baseline = [], []
        for k in range(len(u)):
            current = float(q[k])
            dq = 0.0 if k == 0 else float(q[k] - q[k - 1])
            motor = m[k]
            for j in range(k, min(k + 8, len(u))):
                nextmotor = advance(motor, u[j])
                f = features([current, motor, nextmotor, dq], mean, scale)
                nextq = max(0.0, min(1.0, current + float(f @ coef)))
                errors.append(nextq - q[j + 1])
                baseline.append(q[k] - q[j + 1])
                dq = nextq - current
                current = nextq
                motor = nextmotor
        values = [
            float(np.sqrt(np.mean((prediction[start : start + len(u)] - q[1:]) ** 2))),
            float(np.sqrt(np.mean(np.diff(q) ** 2))),
            float(np.sqrt(np.mean(np.square(errors)))),
            float(np.sqrt(np.mean(np.square(baseline)))),
        ]
        keys = ["one_step_rmse", "hold_one_step_rmse", "rollout_8_rmse", "hold_rollout_8_rmse"]
        np.testing.assert_allclose(values, [reported[k] for k in keys], rtol=0, atol=1e-10)
        if entry["split"] == "holdout":
            eligible &= values[0] <= values[1] and values[2] <= values[3]
        start += len(u)
    assert bool(eligible) == fit["offline_performance_eligible"]
    assert all(digest(p) == s for p, s in refs.items())
    out = dict(
        numerically_verified=True,
        offline_performance_eligible=bool(eligible),
        verified_rows=len(y),
        coefficient_max_abs_difference=difference,
        source_sha256=refs,
        physical_qualification=False,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    with (root / "independent_verified.json").open("x") as stream:
        json.dump(out, stream, indent=2)
        stream.write("\n")
    print("verified", len(y), "rows", "coefficient_max_abs_difference", difference, "eligible", eligible)


if __name__ == "__main__":
    main()
