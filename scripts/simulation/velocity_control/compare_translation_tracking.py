"""Verified082/081 pose, sampled velocity and action variation; descriptive only."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/translation_loaded_control_20261025")
PRIOR = Path("outputs/loaded_velocity_control_20261025")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(v):
    return float(np.sqrt(np.mean(np.sum(np.asarray(v) ** 2, axis=1))))


def main():
    config_path = Path("configs/translation_loaded_control_20261025.json")
    config = json.loads(config_path.read_text())
    prior_config = Path("configs/loaded_velocity_control_20261025.json")
    assert config["profiles"] == json.loads(prior_config.read_text())["profiles"]
    hashes = {str(p): digest(p) for p in [config_path, prior_config, Path(__file__)]}
    results = []
    for i, profile in enumerate(config["profiles"]):
        completed = ROOT / "verification" / f"case{i:02d}" / "exit.json"
        if (
            not completed.exists()
            or json.loads(completed.read_text())["returncode"] != 0
            or profile["format"] != "native_endpoints"
        ):
            continue
        source = Path(profile["trajectory_ref"]["path"])
        assert digest(source) == profile["trajectory_ref"]["sha256"]
        hashes[str(source)] = digest(source)
        with np.load(source, allow_pickle=False) as data:
            p = data["controller_site_pose"][:, :3, 3].copy()
            sourcep = data["source_pose"][:, :3, 3]
            u = data["sent_commands"][:, -1]
            np.testing.assert_allclose(p, sourcep, rtol=0, atol=1e-10)
            native_close = int(np.flatnonzero(u > 0)[0])
            release = int(np.flatnonzero((u < 0) & (np.arange(len(u)) > native_close))[0])
        reference_velocity = (p[2:] - p[:-2]) / 0.1
        variants = {}
        for tag, parent in [("082", ROOT), ("081", PRIOR)]:
            folder = parent / f"case{i:02d}"
            verification = folder / "independent_verified.json"
            v = json.loads(verification.read_text())
            assert v["mechanical_labels_images_passed"]
            log = next(folder.glob("simulation/*/*/steps.jsonl"))
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            for path in [verification, log]:
                hashes[str(path)] = digest(path)
            assert len(rows) == len(p) - 1
            velocity = np.array([r["measured_twist"][:3] for r in rows])[1:]
            action = np.array([r["command"][:3] for r in rows])
            change = np.diff(action, axis=0)
            after = np.array([r["after_source"] for r in rows])[:, :3, 3]
            error = after - p[1:]
            windows = {}
            for label, a, b in [
                ("all", 0, len(rows) - 1),
                ("native_release_window", max(0, release - 5), min(len(rows) - 1, release + 15)),
            ]:
                windows[label] = dict(
                    cycles=[a, b],
                    source_position_rmse_m=rms(error[a:b]),
                    sampled_velocity_error_rmse_m_per_s=rms(velocity[a:b] - reference_velocity[a:b]),
                    normalized_command_increment_rms=rms(change[a:b]),
                )
            variants[tag] = dict(
                stable=v["control"]["outcome"]["stable"],
                outcome=v["control"]["outcome"],
                whole_trajectory_admitted_position_rmse_m=v["control"]["tracking"]["admitted"]["position_rmse_m"],
                windows=windows,
            )
        results.append(
            dict(
                case=i,
                reference=profile["trajectory_ref"],
                native_close=native_close,
                native_release=release,
                variants=variants,
            )
        )
        print(
            i,
            "stable082/081",
            variants["082"]["stable"],
            variants["081"]["stable"],
            "position mm",
            *[round(1000 * variants[k]["whole_trajectory_admitted_position_rmse_m"], 3) for k in ("082", "081")],
            "release velocity error m/s",
            *[
                round(variants[k]["windows"]["native_release_window"]["sampled_velocity_error_rmse_m_per_s"], 5)
                for k in ("082", "081")
            ],
        )
    assert all(digest(p) == sha for p, sha in hashes.items())
    output = dict(
        complete_verified_snapshot=True,
        cases=results,
        source_sha256=hashes,
        velocity_reference="central differences of saved20Hz generated-reference positions;not instantaneous native velocity truth",
        evidence="same profiles, retrospective comparison;not unique root cause or counterfactual simulation",
        new_fits=0,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
    )
    out = Path("outputs/translation_tracking_pairs_20261025")
    out.mkdir(exist_ok=True)
    with (out / f"verified_prefix_{len(results):02d}.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
