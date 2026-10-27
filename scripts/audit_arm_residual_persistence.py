"""085 read-only test of one-period causal arm-model residual persistence."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from s2a.artifacts import sha256


def rms(values):
    return float(np.sqrt(np.mean(np.sum(values**2, axis=1))))


def main():
    hashes = {str(Path(__file__).resolve()): sha256(__file__)}

    def read(path):
        hashes[str(path.resolve())] = sha256(path)
        return json.loads(path.read_text())

    root = Path("outputs/pose_velocity_loaded_control_20261025")
    closure = read(root / "closure_verified.json")
    assert closure["complete"] and closure["owner_normal_wait_verified"]
    cases = []
    for i in range(18):
        folder = root / f"case{i:02d}"
        verified = read(folder / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        case = next(folder.glob("simulation/*/*/controller.json")).parent
        controller = read(case / "controller.json")["contract"]
        log = case / "steps.jsonl"
        hashes[str(log.resolve())] = sha256(log)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        dt = 1 / controller["control_freq"]
        assert dt == 0.05
        kp, kd = np.array(controller["kp"])[:3], np.array(controller["kd"])[:3]
        transitions = np.array([expm(np.array([[0.0, 1.0], [-p, -d]]) * dt) for p, d in zip(kp, kd, strict=True)])
        r, b = 1 - transitions[:, 0, 0], transitions[:, 0, 1]
        c, d = -transitions[:, 1, 0], transitions[:, 1, 1]
        u = np.array([row["command"][:3] for row in rows])
        lo, hi = np.array(controller["input_min"])[:3], np.array(controller["input_max"])[:3]
        outlo, outhi = np.array(controller["output_min"])[:3], np.array(controller["output_max"])[:3]
        delta = outlo + (u - lo) * (outhi - outlo) / (hi - lo)
        v = np.array([row["measured_twist"][:3] for row in rows])
        actual_motion = np.array(
            [np.array(row["after_source"])[:3, 3] - np.array(row["measured_source"])[:3, 3] for row in rows]
        )
        position_innovation = actual_motion - (r * delta + b * v)
        predicted_v = c * delta + d * v
        np.testing.assert_allclose(predicted_v, [row["predicted_linear_velocity"] for row in rows], rtol=0, atol=1e-10)
        velocity_innovation = v[1:] - predicted_v[:-1]
        metrics = {}
        # At cycle t, innovation t-1 is known from the just-observed endpoint.
        # Compare against zero using identical t=1..n-2 support for both channels.
        for name, innovations in [("position", position_innovation[:-1]), ("linear_velocity", velocity_innovation)]:
            target, previous = innovations[1:], innovations[:-1]
            zero, persistence = rms(target), rms(target - previous)
            metrics[name] = dict(
                samples=len(target),
                zero_prediction_rmse=zero,
                previous_residual_prediction_rmse=persistence,
                persistence_improves=persistence < zero,
                units="m" if name == "position" else "m/s",
            )
        cases.append(dict(case=i, task_stable=verified["control"]["outcome"]["stable"], metrics=metrics))
    assert all(sha256(p) == signature for p, signature in hashes.items())
    result = dict(
        complete=True,
        cases=cases,
        source_sha256=hashes,
        prediction="fixed lag1 residual, no fit; only previous observed endpoint is used",
        evidence="retrospective closed-loop logs; not counterfactual compensation or task success",
        physics_steps=0,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
    )
    output = Path("outputs/arm_residual_persistence_20261025")
    output.mkdir(exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name in ("position", "linear_velocity"):
        print(name, "persistence improves", sum(c["metrics"][name]["persistence_improves"] for c in cases), "/18")
    for i in (3, 10, 12):
        print(json.dumps(cases[i]))


if __name__ == "__main__":
    main()
