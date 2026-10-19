"""Materialize only the precommitted additional episode cohort, read-only."""

import argparse
import json

import numpy as np

from s2a.artifacts import load_config, sha256, write_json
from s2a.libero_simulation import sha256_bytes


def main():
    import h5py

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_holdout_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen cohort")
    target = output / "fixture.json"
    if target.exists():
        raise FileExistsError(target)
    records = []
    with h5py.File(config["source_hdf5"]) as source:
        for trial, name in enumerate(config["selected_demos"]):
            demo = source["data"][name]
            initial = np.asarray(demo["states"][0], dtype=float)
            records.append(
                dict(
                    trial=trial,
                    episode="hdf5:" + name,
                    demo=name,
                    initial_state=initial,
                    initial_state_sha256=sha256_bytes(initial.tobytes()),
                    model_xml=demo.attrs["model_file"],
                    action_rows=len(demo["actions"]),
                )
            )
    write_json(
        target,
        dict(
            source_hdf5=config["source_hdf5"],
            source_hdf5_sha256=sha256(config["source_hdf5"]),
            episodes=records,
            selection=config["selection"],
            seed=config["seed"],
        ),
    )
    print(json.dumps(dict(episodes=len(records), fixture=str(target), sha256=sha256(target))))


if __name__ == "__main__":
    main()
