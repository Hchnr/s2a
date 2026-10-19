"""Run one isolated LIBERO trial from the frozen S2A experiment config."""

import argparse

from s2a.artifacts import load_config
from s2a.libero_simulation import run_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_simulation.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("probe", "native", "oracle", "prediction_replay", "policy"), required=True)
    parser.add_argument("--candidate", choices=("baseline", "compensated"), required=True)
    parser.add_argument("--trial", type=int, choices=range(5), required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen experiment")
    run_trial(config, output, stage=args.stage, candidate=args.candidate, trial=args.trial, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
