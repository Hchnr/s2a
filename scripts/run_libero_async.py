"""Execute isolated continuous-control timing probes; no paused inference clock."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from s2a.artifacts import load_config, write_json
from s2a.libero_async import case_name, run_async_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_followup_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=["synthetic", "live"], required=True)
    parser.add_argument("--trial", type=int)
    parser.add_argument("--arm", choices=["affine", "response"], default="affine")
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    c, output = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs from frozen config")
    if args.trial is not None:
        print(
            json.dumps(run_async_case(c, output, stage=args.stage, arm=args.arm, trial=args.trial, delay=args.delay)),
            flush=True,
        )
        return
    cases = (
        [
            (arm, trial, delay)
            for arm in c["synthetic_arms"]
            for trial in c["trials"]
            for delay in c["synthetic_delays_s"]
        ]
        if args.stage == "synthetic"
        else [("affine", trial, 0) for trial in c["trials"]]
    )
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": "",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "LP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def execute(case):
        arm, trial, delay = case
        name = case_name(args.stage, arm, delay)
        log = output / f"{name}_trial{trial}.log"
        if log.exists():
            raise FileExistsError(f"Archive existing evidence before rerunning: {log}")
        command = [
            str(Path(c["simulation_python"]).absolute()),
            __file__,
            "--config",
            args.config,
            "--seed",
            str(args.seed),
            "--stage",
            args.stage,
            "--arm",
            arm,
            "--trial",
            str(trial),
            "--delay",
            str(delay),
        ]
        with log.open("w") as stream:
            code = subprocess.call(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        result = dict(
            arm=arm, trial=trial, delay=delay, exit_code=code, directory=str(output / "async" / name / f"trial{trial}")
        )
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers if args.stage == "synthetic" else 1) as executor:
        results = list(executor.map(execute, cases))
    write_json(
        output / f"matrix_{args.stage}.json", dict(cases=results, complete=all(r["exit_code"] == 0 for r in results))
    )
    if any(r["exit_code"] for r in results):
        raise SystemExit("Some cases failed; all evidence was retained")


if __name__ == "__main__":
    main()
