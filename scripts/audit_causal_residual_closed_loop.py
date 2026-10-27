"""Read-only085 on-policy prediction audit; lower prediction error is not control success."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import sha256


def rms(x):
    return float(np.sqrt(np.mean(np.sum(x * x, axis=1))))


def main():
    root = Path("outputs/causal_residual_loaded_control_20261025")
    hashes = {str(Path(__file__).resolve()): sha256(__file__)}
    cases = []
    for i in range(18):
        receipt = root / "verification" / f"case{i:02d}" / "exit.json"
        if not receipt.exists():
            continue
        hashes[str(receipt.resolve())] = sha256(receipt)
        assert json.loads(receipt.read_text())["returncode"] == 0
        folder = root / f"case{i:02d}"
        verification = folder / "independent_verified.json"
        hashes[str(verification.resolve())] = sha256(verification)
        verified = json.loads(verification.read_text())
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        path = next(folder.glob("simulation/*/*/steps.jsonl"))
        hashes[str(path.resolve())] = sha256(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        actual_motion = np.array(
            [np.array(r["after_source"])[:3, 3] - np.array(r["measured_source"])[:3, 3] for r in rows]
        )[1:-1]
        nominal_motion = np.array([r["nominal_predicted_motion"][:3] for r in rows])[1:-1]
        bias = np.array([r["response_residual"]["position"] for r in rows])[1:-1]
        v = np.array([r["measured_twist"][:3] for r in rows])[2:]
        nominal_v = np.array([r["nominal_predicted_linear_velocity"] for r in rows])[1:-1]
        corrected_v = np.array([r["predicted_linear_velocity"] for r in rows])[1:-1]
        metrics = {}
        for label, y, nominal, corrected in [
            ("position", actual_motion, nominal_motion, nominal_motion + bias),
            ("linear_velocity", v, nominal_v, corrected_v),
        ]:
            metrics[label] = dict(
                samples=len(y),
                nominal_prediction_error_rmse=rms(y - nominal),
                corrected_prediction_error_rmse=rms(y - corrected),
                corrected_is_better=rms(y - corrected) < rms(y - nominal),
            )
        cases.append(
            dict(
                case=i,
                performance_qualified=verified["performance_qualified"],
                task_stable=verified["control"]["outcome"]["stable"],
                saturated_cycles=sum(r["saturated"] for r in rows),
                rate_limited_cycles=sum(r["rate_limited"] for r in rows),
                metrics=metrics,
            )
        )
    assert cases and all(sha256(p) == h for p, h in hashes.items())
    out = Path("outputs/causal_residual_on_policy_audit_20261025")
    out.mkdir(exist_ok=True)
    with (out / f"verified_prefix_{len(cases):02d}.json").open("x") as stream:
        json.dump(
            dict(
                complete_verified_snapshot=True,
                cases=cases,
                source_sha256=hashes,
                physics_steps=0,
                new_fits=0,
                model_calls=0,
                fresh_cases=0,
                evidence="actual085 logs; retrospective prediction only; no counterfactual rollout",
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    for c in cases:
        if c["case"] in (3, 11, 14):
            print(json.dumps(c))
    print(
        "verified",
        len(cases),
        "prediction improved counts",
        {k: sum(c["metrics"][k]["corrected_is_better"] for c in cases) for k in ("position", "linear_velocity")},
    )


if __name__ == "__main__":
    main()
