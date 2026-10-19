"""Run fresh-endpoint WAM/oracle comparisons with inference latency excluded."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from s2a.artifacts import load_config, write_json
from s2a.libero_effect import run_effect_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_wam_effect_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=["controls", "wam"], required=True)
    parser.add_argument("--case")
    parser.add_argument("--trial", type=int)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    c, out = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs from frozen experiment")
    cases = [r for r in c["effect_cases"] if r["source"] == ("oracle" if args.stage == "controls" else "wam")]
    if args.case:
        case = next(r for r in cases if r["name"] == args.case)
        print(json.dumps(run_effect_case(c, out, case, args.trial)), flush=True)
        return
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": "",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "LP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def execute_trial(trial):
        results = []
        for case in cases:
            path = out / f"{case['name']}_trial{trial}.log"
            if path.exists():
                raise FileExistsError(f"archive existing evidence before retry: {path}")
            command = [
                str(Path(c["simulation_python"]).absolute()),
                __file__,
                "--config",
                args.config,
                "--seed",
                str(args.seed),
                "--stage",
                args.stage,
                "--case",
                case["name"],
                "--trial",
                str(trial),
            ]
            with path.open("w") as stream:
                code = subprocess.call(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            row = dict(
                case=case, trial=trial, exit_code=code, directory=str(out / "sync" / case["name"] / f"trial{trial}")
            )
            results.append(row)
            print(json.dumps(row), flush=True)
        return results

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = [row for group in executor.map(execute_trial, c["trials"]) for row in group]
    write_json(
        out / f"matrix_{args.stage}.json", dict(cases=results, complete=all(r["exit_code"] == 0 for r in results))
    )
    if any(r["exit_code"] for r in results):
        raise SystemExit("some effect cases failed; all evidence retained")


if __name__ == "__main__":
    main()
