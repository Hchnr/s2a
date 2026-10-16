"""Run the read-only DROID study from the S2A repository root."""

import argparse

from s2a.analytic import analytic
from s2a.artifacts import load_config
from s2a.data import extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/droid_episode0.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=["extract", "analytic"], default="analytic")
    args = parser.parse_args()
    config, out = load_config(args.config)
    if args.seed != config["seed"]:
        parser.error("--seed must match the saved experiment configuration")
    arrays, audit = extract(config, out)
    print(f"P0: {audit['selected_transitions']} transitions; artifacts: {out}")
    if args.stage == "analytic":
        metrics = analytic(config, arrays, out)
        print(f"P1 geometry gate: {metrics['geometry_passed']}")


if __name__ == "__main__":
    main()
