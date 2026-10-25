"""Read-only verified083/082 tracking comparison; no causal or task gate claims."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import sha256

ROOT = Path("outputs/pose_velocity_loaded_control_20261025")
PRIOR = Path("outputs/translation_loaded_control_20261025")


def main():
    hashes = {}

    def read(path):
        hashes[str(path.resolve())] = sha256(path)
        return json.loads(path.read_text())

    current = read(Path("configs/pose_velocity_loaded_control_20261025.json"))
    previous = read(Path("configs/translation_loaded_control_20261025.json"))
    assert current["profiles"] == previous["profiles"]
    cases = []
    for i in range(18):
        receipt = ROOT / "verification" / f"case{i:02d}" / "exit.json"
        if not receipt.exists() or read(receipt)["returncode"] != 0:
            continue
        variants = {}
        logs = {}
        for tag, root in [("083", ROOT), ("082", PRIOR)]:
            folder = root / f"case{i:02d}"
            v = read(folder / "independent_verified.json")
            assert v["complete"] and v["mechanical_labels_images_passed"]
            log = next(folder.glob("simulation/*/*/steps.jsonl"))
            hashes[str(log.resolve())] = sha256(log)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            logs[tag] = rows
            variants[tag] = dict(
                qualified=v["performance_qualified"],
                outcome=v["control"]["outcome"],
                tracking=v["control"]["tracking"]["admitted"],
            )
        a, b = logs["083"], logs["082"]
        assert len(a) == len(b)
        np.testing.assert_allclose(
            [x["effective_desired_tcp"] for x in a], [x["effective_desired_tcp"] for x in b], atol=1e-10, rtol=0
        )
        # The next measured pre-step twist is the actual previous endpoint twist.
        w = np.array([x["linear_velocity_reference"]["velocity"] for x in a[:-1]])
        for tag, rows in logs.items():
            v = np.array([x["measured_twist"][:3] for x in rows[1:]])
            u = np.array([x["command"][:3] for x in rows])
            variants[tag]["sampled_velocity_error_rmse_m_per_s"] = float(np.sqrt(np.mean(np.sum((v - w) ** 2, axis=1))))
            variants[tag]["normalized_command_increment_rms"] = float(
                np.sqrt(np.mean(np.sum(np.diff(u, axis=0) ** 2, axis=1)))
            )
        cases.append(dict(case=i, variants=variants, endpoint_velocity_samples=len(w)))
    assert cases
    assert all(sha256(path) == signature for path, signature in hashes.items())
    output = Path("outputs/pose_velocity_tracking_pairs_20261025")
    output.mkdir(exist_ok=True)
    result = dict(
        verified_snapshot=True,
        cases=cases,
        source_sha256=hashes,
        velocity_reference="083 generated-reference difference estimate; not native instantaneous velocity truth",
        evidence="retrospective paired logs; no unique causal attribution",
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (output / f"verified_prefix_{len(cases):02d}.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
