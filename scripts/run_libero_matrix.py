"""Execute paired trials in subprocesses; each trial owns one policy replica."""

import argparse
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from s2a.artifacts import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_simulation.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("qualification", "prediction_replay", "policy"), required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--server-timeout", type=float, default=5400)
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"] or args.workers < 1:
        raise ValueError("invalid seed or worker count")
    if args.stage == "policy":
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        started = time.monotonic()
        while True:
            ready = []
            for url in config["server_urls"]:
                try:
                    with opener.open(url + "/healthz", timeout=2) as response:
                        ready.append(json.load(response).get("status") == "ready")
                except OSError:
                    ready.append(False)
            write_json(output / "policy_readiness.json", dict(ready=ready, waited_s=time.monotonic() - started))
            if all(ready):
                break
            if time.monotonic() - started > args.server_timeout:
                raise TimeoutError("Policy replicas did not become ready; see server status/logs")
            time.sleep(10)
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": "",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "LP_NUM_THREADS": "2",
        "MPLCONFIGDIR": str(output / "matplotlib"),
    }

    def execute_trial(trial):
        stages = ("native", "oracle") if args.stage == "qualification" else (args.stage,)
        results = []
        for stage in stages:
            candidates = ("baseline",) if stage == "native" else ("baseline", "compensated")
            if args.stage == "qualification" and trial == 0:
                stages_probe = [("probe", c) for c in ("baseline", "compensated")]
            else:
                stages_probe = []
            jobs = (stages_probe if stage == "native" else []) + [(stage, c) for c in candidates]
            for actual_stage, candidate in jobs:
                dest = output / actual_stage / candidate / f"trial{trial}"
                if (dest / "result.json").is_file():
                    saved = json.loads((dest / "manifest.json").read_text())
                    expected = {**config, "server_url": config["server_urls"][trial]}
                    if saved["config"] != expected:
                        raise ValueError(f"Existing result has different frozen config: {dest}")
                    result = json.loads((dest / "result.json").read_text())
                    if result["error"]:
                        raise ValueError(f"Existing trial failed: {dest}")
                    results.append(result)
                    continue
                command = [
                    str(Path.cwd() / "outputs/libero_env311/bin/python"),
                    "scripts/run_libero_simulation.py",
                    "--config",
                    args.config,
                    "--seed",
                    str(args.seed),
                    "--stage",
                    actual_stage,
                    "--candidate",
                    candidate,
                    "--trial",
                    str(trial),
                ]
                log_path = output / f"{actual_stage}_{candidate}_trial{trial}.log"
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    code = process.wait()
                if code:
                    raise RuntimeError(f"Trial failed with exit {code}: {log_path}")
                results.append(json.loads((dest / "result.json").read_text()))
        return results

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute_trial, trial): trial for trial in config["trials"]}
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as exc:
                failures.append(dict(trial=futures[future], error=repr(exc)))
            write_json(output / f"matrix_{args.stage}.json", dict(results=results, failures=failures))
            print(json.dumps(dict(completed_trials=len(results), failures=failures)), flush=True)
    if failures:
        raise RuntimeError(f"Failed trials: {failures}")


if __name__ == "__main__":
    main()
