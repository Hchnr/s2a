"""Prepare the preselected published-state oracle cohort from frozen bridges."""

import argparse
import json
from pathlib import Path

from s2a.artifacts import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_published_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen cohort")
    target = output / "fixture.json"
    if target.exists():
        raise FileExistsError(target)
    original = json.loads(Path("configs/libero_simulation.json").read_text())
    base = json.loads(Path(original["fixture"]).read_text())
    extra = json.loads(Path("outputs/libero_holdout_20261019/fixture.json").read_text())
    records = {row["demo"]: row for row in base["episodes"] + extra["episodes"]}
    trajectories = json.loads(Path(config["published_bridge"]).read_text())["trajectories"]
    if set(config["selected_demos"]) != set(trajectories):
        raise ValueError("published bridge differs from frozen episode selection")
    selected = [
        dict(records[name], trial=trial, episode=f"R2:{trajectories[name]['r2_episode']}/HDF5:{name}")
        for trial, name in enumerate(config["selected_demos"])
    ]
    write_json(target, {**base, "episodes": selected, "selection": config["selection"]})
    print(json.dumps(dict(episodes=len(selected), fixture=str(target))))


if __name__ == "__main__":
    main()
