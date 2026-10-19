"""Run the frozen multi-task qualification or delay-free checkpoint matrix."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from s2a.artifacts import load_config, write_json
from s2a.checkpoint_simulation import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("native", "oracle", "wam", "probe"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--record")
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--probe", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c, out = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    records = json.loads((out / "simulation/fixtures.json").read_text())["records"]
    probes = [
        dict(name=f"p{p:g}_r{r:g}", position_amplitude_m=p, rotation_amplitude_rad=r)
        for p, r in ((0.005, 0), (0.02, 0), (0.05, 0), (0, 0.05), (0, 0.2), (0, 0.5))
    ]
    if args.record:
        record = next(r for r in records if r["name"] == args.record)
        result = run_case(
            c,
            out,
            record,
            source=args.stage,
            sampling_seed=args.sampling_seed,
            checkpoint=args.checkpoint,
            probe=probes[args.probe] if args.probe is not None else None,
        )
        print(json.dumps(result), flush=True)
        return
    if args.stage == "wam" and args.checkpoint is None:
        raise ValueError("select one endpoint per matrix worker")
    jobs = [(record, seed, None) for record in records for seed in c["sampling_seeds"]]
    if args.stage == "probe":
        jobs = [(records[0], c["seed"], i) for i in range(len(probes))]
    env = dict(
        os.environ,
        LD_LIBRARY_PATH="",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        LP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )

    def execute(job):
        record, seed, probe = job
        label = args.checkpoint or args.stage
        if probe is not None:
            label += "_" + probes[probe]["name"]
        directory = out / "simulation" / label / f"{record['name']}_s{seed}"
        if args.resume and (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_text())
            if result["error"] or result["steps"] != result["expected_steps"]:
                raise ValueError("failed evidence must be archived before a retry")
            return dict(directory=str(directory), exit_code=0, resumed=True)
        log = out / "simulation" / f"{label}_{record['name']}_s{seed}.log"
        if log.exists():
            raise FileExistsError(log)
        command = [
            str(Path(c["simulation_python"]).absolute()),
            __file__,
            "--config",
            args.config,
            "--seed",
            str(args.seed),
            "--stage",
            args.stage,
            "--record",
            record["name"],
            "--sampling-seed",
            str(seed),
        ]
        if args.checkpoint:
            command += ["--checkpoint", args.checkpoint]
        if probe is not None:
            command += ["--probe", str(probe)]
        with log.open("w") as stream:
            code = subprocess.call(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        result = dict(directory=str(directory), exit_code=code)
        print(json.dumps(result), flush=True)
        return result

    workers = 1 if args.stage == "wam" else args.workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(execute, jobs))
    write_json(
        out / "simulation" / f"matrix_{args.checkpoint or args.stage}.json",
        dict(cases=results, complete=all(x["exit_code"] == 0 for x in results)),
    )
    if any(x["exit_code"] for x in results):
        raise SystemExit("matrix has failed cases; evidence retained")


if __name__ == "__main__":
    main()
