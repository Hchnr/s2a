"""Post-failure descriptive coverage audit; no parameter fitting or new acceptance gate."""

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base = Path("outputs/all_phase_gripper_forward_20261025")
    velocity = Path("outputs/gripper_velocity_state_audit_20261025")
    paths = [
        base / "parameters.json",
        base / "fit_cache.npz",
        Path("outputs/gripper_forward_identification_20261025/fit_result.json"),
        velocity / "audit.json",
        velocity / "velocity_traces.npz",
        Path(__file__),
    ]
    refs = {str(p): digest(p) for p in paths}
    params = json.loads(paths[0].read_text())
    roster = json.loads(paths[2].read_text())["trajectories"]
    audit = json.loads(paths[3].read_text())
    with np.load(paths[1], allow_pickle=False) as data:
        raw = data["raw"]
        target = data["target_delta_q"]
        fit = data["weights"] > 0
    vmap = {row["ref"]["path"]: i for i, row in enumerate(audit["cases"])}
    locations = []
    vel = []
    with np.load(paths[4], allow_pickle=False) as traces:
        for entry in roster:
            values = traces[f"case{vmap[entry['ref']['path']]:02d}_velocity"][:-1] * 0.05
            locations.extend(dict(path=entry["ref"]["path"], cycle=k) for k in range(len(values)))
            vel.extend(values)
    query = next(i for i, row in enumerate(locations) if "task1_native_demo_39/" in row["path"] and row["cycle"] == 157)
    z = (raw - params["mean"]) / params["scale"]
    distance = np.linalg.norm(z - z[query], axis=1)
    distance[~fit] = np.inf
    nearest = np.argsort(distance)[:5]

    def describe(i):
        return dict(
            **locations[i],
            raw=raw[i].tolist(),
            instantaneous_delta_equivalent=float(vel[i]),
            actual_next_opening_delta=float(target[i]),
        )

    band = (raw[:, 0] < 0.7) & (raw[:, 1] > 0.99) & (raw[:, 2] > 0.99) & (raw[:, 3] > 0) & (raw[:, 3] < 0.07)
    output = dict(
        complete=True,
        post_failure_diagnostic=True,
        not_a_new_gate=True,
        band="q<0.7, current_motor>0.99, next_motor>0.99, 0<previous_delta_q<0.07",
        fit_band_rows=int(np.sum(band & fit)),
        internal_holdout_band_rows=int(np.sum(band & ~fit)),
        query=describe(query),
        nearest_fit=[dict(**describe(int(i)), normalized_distance=float(distance[i])) for i in nearest],
        source_sha256=refs,
        new_fits=0,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    assert all(digest(p) == sha for p, sha in refs.items())
    with (velocity / "coverage_diagnosis.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(
        "post-failure diagnostic band: fit",
        output["fit_band_rows"],
        "internal holdout",
        output["internal_holdout_band_rows"],
    )


if __name__ == "__main__":
    main()
