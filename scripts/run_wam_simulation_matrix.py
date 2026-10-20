"""Run the frozen multi-task qualification or delay-free checkpoint matrix."""

import argparse
import json
import os
import subprocess
import time
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
    parser.add_argument("--server-port", type=int, help="separate replica of the same verified frozen weights")
    parser.add_argument("--record")
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--probe", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--job-indices", help="disjoint indices into the frozen initial-state/seed matrix")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-state", action="store_true", help="serialize work on a single-request GPU server")
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
            server_port=args.server_port,
        )
        print(json.dumps(result), flush=True)
        return
    if args.stage == "wam" and args.checkpoint is None:
        raise ValueError("select one endpoint per matrix worker")
    if args.wait_for_state:
        if args.stage != "wam":
            raise ValueError("only WAM jobs wait for the shared model endpoint")
        completion = out / "state_queries" / args.checkpoint / "complete.json"
        while not completion.exists():
            print(json.dumps(dict(waiting_for=str(completion))), flush=True)
            time.sleep(30)
        expected = (
            len(c["tasks"])
            * c["offline_episodes_per_task"]
            * 4
            * len(c["offline_horizons"])
            * len(c["offline_prompt_modes"])
            * len(c["sampling_seeds"])
        )
        if json.loads(completion.read_text())["cases"] != expected:
            raise ValueError("offline matrix did not complete")
    jobs = [(record, seed, None) for record in records for seed in c["sampling_seeds"]]
    if args.stage == "probe":
        jobs = [(records[0], c["seed"], i) for i in range(len(probes))]
    if args.job_indices:
        indices = [int(value) for value in args.job_indices.split(",")]
        if len(indices) != len(set(indices)) or any(i < 0 or i >= len(jobs) for i in indices):
            raise ValueError("invalid or duplicate job index")
        jobs = [jobs[i] for i in indices]
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
            manifest = json.loads((directory / "manifest.json").read_text())
            expected = dict(
                seed=seed,
                task_id=record["task_id"],
                task_text=record["task"],
                max_steps=c["online_max_steps"],
                action_chunk_size=c["online_chunk_size"],
                execution_horizon=c["online_execution_horizon"],
                **{
                    k: c[k]
                    for k in (
                        "sampling_steps",
                        "guidance",
                        "shift",
                        "control_fps",
                        "inverse_weight",
                        "max_command_rate",
                        "opening_hysteresis",
                    )
                },
            )
            if any(manifest["config"].get(k) != value for k, value in expected.items()):
                raise ValueError("cannot resume a changed policy/controller protocol")
            if manifest["record"]["initial_state_sha256"] != record["initial_state_sha256"]:
                raise ValueError("cannot resume a changed initial state")
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
        if args.server_port:
            command += ["--server-port", str(args.server_port)]
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
    suffix = f"_jobs{args.job_indices.replace(',', '-')}" if args.job_indices else ""
    write_json(
        out / "simulation" / f"matrix_{args.checkpoint or args.stage}{suffix}.json",
        dict(cases=results, complete=all(x["exit_code"] == 0 for x in results)),
    )
    if any(x["exit_code"] for x in results):
        raise SystemExit("matrix has failed cases; evidence retained")


if __name__ == "__main__":
    main()
