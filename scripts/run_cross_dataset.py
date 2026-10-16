"""Evaluate the frozen same-arm cohort; no robot commands are executed."""

import argparse

from s2a.artifacts import load_config
from s2a.cross_dataset import run_cross_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cross_dataset_franka.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config, out = load_config(args.config)
    if args.seed != config["seed"]:
        parser.error("seed differs from frozen cohort")
    result = run_cross_dataset(config, out)
    failed = sum(c["selected_episodes"] - c["geometry_passed_episodes"] for c in result["cohorts"])
    print(f"Report: {out / 'report.md'}; nonpassing episodes: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
