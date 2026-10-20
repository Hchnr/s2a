"""Run isolated CPU physics workers against one frozen WAM service/variant."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from s2a.artifacts import load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--source", choices=("wam", "oracle"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--record")
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--job-indices")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    c, root = load_config(args.config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("both coordinator and physics require empty CUDA and LD paths")
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    if args.variant not in {v["name"] for v in c["variants"]}:
        raise ValueError("unknown variant")
    if args.source == "wam":
        ck = next(ck for ck in c["checkpoints"] if ck["name"] == args.checkpoint)
        if ck["gpu"] not in (4, 5, 6, 7):
            raise ValueError("forbidden physical GPU")
    elif args.checkpoint is not None:
        raise ValueError("oracle qualification cannot be labeled as a WAM checkpoint")
    if sha256(c["fixture"]) != c["fixture_sha256"]:
        raise ValueError("changed fixture file")
    records = json.loads(Path(c["fixture"]).read_text())["records"]
    selected = [r for r in records if r["name"] in c["records"]]
    if len(selected) != len(c["records"]):
        raise ValueError("missing fixture record")
    if args.record:
        from s2a.stability_simulation import run_stability_case

        record = next(r for r in selected if r["name"] == args.record)
        if args.sampling_seed not in c["sampling_seeds"]:
            raise ValueError("sampling seed outside the protocol")
        result = run_stability_case(
            c,
            root,
            record,
            source=args.source,
            variant_name=args.variant,
            sampling_seed=args.sampling_seed,
            checkpoint=args.checkpoint,
        )
        print(json.dumps(result), flush=True)
        return
    jobs = [(r, seed) for r in selected for seed in c["sampling_seeds"]]
    if args.job_indices:
        indices = [int(index) for index in args.job_indices.split(",")]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(jobs) for i in indices):
            raise ValueError("invalid job selection")
        jobs = [jobs[i] for i in indices]
    if args.dry_run:
        print(json.dumps(dict(jobs=[(r["name"], s) for r, s in jobs], source=args.source, variant=args.variant)))
        return
    label = f"{args.checkpoint or args.source}_{args.variant}"
    logs = root / "simulation/logs" / label
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        LP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )

    def execute(job):
        record, seed = job
        command = [
            str(Path(c["simulation_python"]).absolute()),
            __file__,
            "--config",
            args.config,
            "--seed",
            str(args.seed),
            "--source",
            args.source,
            "--variant",
            args.variant,
            "--record",
            record["name"],
            "--sampling-seed",
            str(seed),
        ]
        if args.checkpoint:
            command += ["--checkpoint", args.checkpoint]
        log = logs / f"{record['name']}_s{seed}.log"
        with log.open("x") as stream:
            code = subprocess.call(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        result = dict(record=record["name"], sampling_seed=seed, returncode=code, command=command)
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=1 if args.source == "wam" else args.workers) as pool:
        results = list(pool.map(execute, jobs))
    suffix = f"_jobs{args.job_indices.replace(',', '-')}" if args.job_indices else ""
    write_json(
        root / "simulation" / f"matrix_{label}{suffix}.json",
        dict(complete=all(not r["returncode"] for r in results), cases=results),
    )
    if any(r["returncode"] for r in results):
        raise SystemExit("physics matrix failed; original evidence retained")


if __name__ == "__main__":
    main()
