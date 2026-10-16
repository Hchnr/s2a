"""Run the read-only DROID study from the S2A repository root."""

import argparse

from s2a.analytic import analytic
from s2a.artifacts import load_config
from s2a.data import extract
from s2a.reporting import plot_results, render_report
from s2a.rollout import compare_controllers
from s2a.study import calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/droid_episode0.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=["extract", "analytic", "calibrate", "all", "report"], default="all")
    parser.add_argument("--plots", action="store_true", help="Generate figures (requires .[report])")
    args = parser.parse_args()
    config, out = load_config(args.config)
    if args.seed != config["seed"]:
        parser.error("--seed must match the saved experiment configuration")
    if args.stage == "report":
        if args.plots:
            plot_results(out)
        render_report(out)
        print(f"Report: {out / 'report.md'}")
        return
    arrays, audit = extract(config, out)
    print(f"P0: {audit['selected_transitions']} transitions; artifacts: {out}")
    if args.stage in {"analytic", "calibrate", "all"}:
        metrics = analytic(config, arrays, out)
        print(f"P1 geometry gate: {metrics['geometry_passed']}")

    if args.stage in {"calibrate", "all"}:
        model, params, _ = calibrate(config, arrays, out)
        print(f"P2 response gate: {params['response_gate_passed']}; {params['gates']}")

    if args.stage == "all":
        result = compare_controllers(config, arrays, model, params, out)
        print(f"P3 local model rollout executed: {result['executed']}")
        if args.plots:
            plot_results(out)
        render_report(out)


if __name__ == "__main__":
    main()
