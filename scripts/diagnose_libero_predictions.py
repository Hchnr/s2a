"""Offline first-chunk diagnostics; labels never enter the running policy."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import load_config, sha256, write_json
from s2a.geometry import rot6d_to_matrix
from s2a.libero_simulation import expert_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_simulation.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen config")
    fixture = json.loads(Path(config["fixture"]).read_text())
    h = np.asarray(json.loads(Path(config["tool_contract"]).read_text())["source_from_tcp"])
    rows = []
    for trial in config["trials"]:
        path = output / "policy/baseline" / f"trial{trial}" / "response_0000.json"
        predicted = np.asarray(json.loads(path.read_text())["outputs"][0]["content"]["action"])
        target, _, _, _ = expert_reference(fixture, fixture["episodes"][trial], h)
        target = target[: len(predicted)]
        rotations = rot6d_to_matrix(predicted[:, 3:9])
        expected = rot6d_to_matrix(target[:, 3:9])
        errors = Rotation.from_matrix(rotations @ expected.swapaxes(-1, -2)).magnitude()
        rows.append(
            dict(
                trial=trial,
                episode=config["episodes"][trial],
                prediction_file_sha256=sha256(path),
                frames=len(predicted),
                translation_rmse_per_axis_m=float(np.sqrt(np.mean((predicted[:, :3] - target[:, :3]) ** 2))),
                rotation_geodesic_rmse_rad=float(np.sqrt(np.mean(errors**2))),
                first8_predicted_rotation_mean_rad=float(np.mean(Rotation.from_matrix(rotations[:8]).magnitude())),
                first8_expert_rotation_mean_rad=float(np.mean(Rotation.from_matrix(expected[:8]).magnitude())),
                raw_opening_min=float(predicted[:, 9].min()),
                raw_opening_max=float(predicted[:, 9].max()),
                raw_opening_outside_fraction=float(np.mean((predicted[:, 9] < 0) | (predicted[:, 9] > 1))),
                opening_mae_before_clip=float(np.mean(abs(predicted[:, 9] - target[:, 9]))),
                opening_mae_after_clip=float(np.mean(abs(np.clip(predicted[:, 9], 0, 1) - target[:, 9]))),
            )
        )
    write_json(
        output / "initial_prediction_diagnostics.json",
        dict(
            evidence="Offline first chunk; same initial image as live evaluation, expert states used only as diagnostic labels.",
            checkpoint_iteration=config["checkpoint_iteration"],
            source_hdf5_sha256=sha256(fixture["source_hdf5"]),
            cases=rows,
            mean={
                k: float(np.mean([r[k] for r in rows]))
                for k in rows[0]
                if k not in ("trial", "episode", "prediction_file_sha256")
            },
            tuning_performed=False,
            held_out=False,
        ),
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
