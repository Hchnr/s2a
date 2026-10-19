"""Run isolated pre/post timing, gripper and OSC factorial diagnostics."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from s2a.artifacts import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_audit_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case")
    parser.add_argument("--trial", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen config")
    if args.case:
        from s2a.libero_audit import run_audit_case

        case = next(c for c in config["cases"] if c["name"] == args.case)
        print(json.dumps(run_audit_case(config, output, case, args.trial)), flush=True)
        return
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": "",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "LP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def execute(case, trial):
        dest = output / case["name"] / f"trial{trial}"
        if dest.exists():
            if not args.resume or not (dest / "result.json").exists():
                raise FileExistsError(f"Archive unfinished evidence before rerunning {dest}")
            manifest = json.loads((dest / "manifest.json").read_text())
            result = json.loads((dest / "result.json").read_text())
            fixture = json.loads(Path(config["simulation"]["fixture"]).read_text())
            reference_length = (
                config["reference_steps"]
                if config.get("reference_source") == "saved_policy"
                else fixture["episodes"][trial]["action_rows"]
            )
            length = reference_length + (
                config.get("native_extra_hold_steps", 0) if case["arm"] == "native" else config["extra_hold_steps"]
            )
            if (
                manifest["config"] != config
                or manifest["case"] != case
                or result["trial"] != trial
                or result["error"] is not None
                or result["steps"] < 1
                or (not config.get("stop_on_success", True) and result["steps"] != length)
                or (config.get("stop_on_success", True) and not result["success"] and result["steps"] != length)
            ):
                raise ValueError(f"Existing evidence does not match the frozen completed case: {dest}")
            return result
        log_path = output / f"{case['name']}_trial{trial}.log"
        command = [
            sys.executable,
            __file__,
            "--config",
            args.config,
            "--seed",
            str(args.seed),
            "--case",
            case["name"],
            "--trial",
            str(trial),
        ]
        with log_path.open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"case failed; see {log_path}")
        return json.loads((dest / "result.json").read_text())

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(execute, case, trial): (case["name"], trial)
            for case in config["cases"]
            for trial in config["trials"]
        }
        for future in as_completed(jobs):
            try:
                row = future.result()
                results.append(row)
                print(json.dumps(row), flush=True)
            except Exception as exc:
                failures.append(dict(case=jobs[future], error=repr(exc)))
            write_json(output / "matrix.json", dict(results=results, failures=failures))
    if failures:
        raise RuntimeError(f"failed jobs: {failures}")


if __name__ == "__main__":
    main()
