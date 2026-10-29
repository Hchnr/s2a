"""075 one fixed quadratic state-response fit and free q rollout audit."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/all_phase_gripper_forward_20261025")
ROSTER = Path("outputs/gripper_forward_identification_20261025/fit_result.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motor_step(motor, command):
    for _ in range(25):
        motor = float(np.clip(motor - 0.01 * np.sign(command), -1, 1))
    return motor


def design(raw, mean, scale):
    z = (np.asarray(raw) - mean) / scale
    return np.concatenate(
        [
            z,
            np.stack([z[..., i] * z[..., j] for i in range(4) for j in range(i, 4)], axis=-1),
            np.ones(z.shape[:-1] + (1,)),
        ],
        axis=-1,
    )


def main():
    paths = [ROSTER, Path(__file__), Path("plans/v0.0.1/075_all_phase_gripper_forward_identification_20261025.md")]
    roster = json.loads(ROSTER.read_text())["trajectories"]
    paths += [Path(d["ref"]["path"]) for d in roster]
    hashes = {str(p): digest(p) for p in paths}
    raw, targets, datasets = [], [], []
    for entry in roster:
        assert digest(entry["ref"]["path"]) == entry["ref"]["sha256"]
        with np.load(entry["ref"]["path"], allow_pickle=False) as data:
            q = data["opening"].copy()
            commands = data["sent_commands"][:, -1].copy()
            offsets, flat = data["gripper_current_action_offsets"], data["gripper_current_action_flat"]
            motors = np.array([flat[offsets[k]] for k in range(len(q))])
        assert len(commands) == len(q) - 1
        start = len(raw)
        for k, u in enumerate(commands):
            mnext = motor_step(motors[k], u)
            np.testing.assert_allclose(mnext, motors[k + 1], rtol=0, atol=1e-12)
            raw.append([q[k], motors[k], mnext, 0.0 if k == 0 else q[k] - q[k - 1]])
            targets.append(q[k + 1] - q[k])
        datasets.append(dict(entry=entry, start=start, stop=len(raw), q=q, commands=commands, motors=motors))
    raw, y = np.array(raw), np.array(targets)
    w = np.zeros(len(y))
    for d in datasets:
        if d["entry"]["split"] == "fit":
            w[d["start"] : d["stop"]] = 1 / (24 * (d["stop"] - d["start"]))
    assert sum(d["entry"]["split"] == "fit" for d in datasets) == 24
    np.testing.assert_allclose(w.sum(), 1, rtol=0, atol=1e-12)
    mean = (w[:, None] * raw).sum(axis=0)
    scale = np.maximum(np.sqrt((w[:, None] * (raw - mean) ** 2).sum(axis=0)), 1e-8)
    x = design(raw, mean, scale)
    penalty = np.eye(15) * 0.001
    penalty[-1, -1] = 0
    coef = np.linalg.solve(x.T @ (w[:, None] * x) + penalty, x.T @ (w * y))
    predicted = np.clip(raw[:, 0] + x @ coef, 0, 1)
    metrics = []
    for d in datasets:
        q, u, m = d["q"], d["commands"], d["motors"]
        sl = slice(d["start"], d["stop"])
        errors, holds = [], []
        for k in range(len(u)):
            current = float(q[k])
            dq = 0.0 if k == 0 else float(q[k] - q[k - 1])
            motor = float(m[k])
            for j in range(k, min(k + 8, len(u))):
                nextmotor = motor_step(motor, u[j])
                features = design(np.array([current, motor, nextmotor, dq]), mean, scale)
                nextq = float(np.clip(current + features @ coef, 0, 1))
                errors.append(nextq - q[j + 1])
                holds.append(q[k] - q[j + 1])
                dq = nextq - current
                current = nextq
                motor = nextmotor
        metrics.append(
            dict(
                ref=d["entry"]["ref"],
                split=d["entry"]["split"],
                task=d["entry"]["task"],
                one_step_rmse=float(np.sqrt(np.mean((predicted[sl] - q[1:]) ** 2))),
                hold_one_step_rmse=float(np.sqrt(np.mean(np.diff(q) ** 2))),
                rollout_8_rmse=float(np.sqrt(np.mean(np.square(errors)))),
                hold_rollout_8_rmse=float(np.sqrt(np.mean(np.square(holds)))),
            )
        )
    held = [d for d in metrics if d["split"] == "holdout"]
    eligible = all(
        d["one_step_rmse"] <= d["hold_one_step_rmse"] and d["rollout_8_rmse"] <= d["hold_rollout_8_rmse"] for d in held
    )
    assert all(digest(p) == s for p, s in hashes.items())
    ROOT.mkdir(exist_ok=False)
    np.savez(ROOT / "fit_cache.npz", raw=raw, target_delta_q=y, weights=w)
    params = dict(
        seed=42, ridge=0.001, mean=mean.tolist(), scale=scale.tolist(), coefficients=coef.tolist(), feature_dimension=15
    )
    (ROOT / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    result = dict(
        complete=True,
        offline_performance_eligible=eligible,
        physical_qualification=False,
        source_sha256=hashes,
        metrics=metrics,
        parameters_sha256=digest(ROOT / "parameters.json"),
        cache_sha256=digest(ROOT / "fit_cache.npz"),
        new_fits=1,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    (ROOT / "fit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(eligible=eligible, holdout=held), indent=2))


if __name__ == "__main__":
    main()
