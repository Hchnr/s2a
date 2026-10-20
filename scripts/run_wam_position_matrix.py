"""CPU-only isolated position/hold trials against an already owned WAM service.

No model is launched here. Each service has at most one physical worker;
oracle trajectories are explicitly privileged qualifications. This independent
runner leaves the live stability matrix and its protocol bytes untouched.
"""

import argparse
import fcntl
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from s2a.artifacts import REPO, sha256, write_json


def protocol(config_path, *, source, variant_name, checkpoint):
    """Validate without creating any output directory, including --dry-run."""
    from s2a.stability_position_simulation import position_variant

    path = Path(config_path).resolve()
    if not path.is_relative_to(REPO):
        raise ValueError("protocol must be inside this repository")
    c = json.loads(path.read_text())
    root = (REPO / c["output_dir"]).resolve()
    if not root.is_relative_to(REPO / "outputs") or root == REPO / "outputs":
        raise ValueError("output_dir must be a named repository outputs subdirectory")
    if c["seed"] != 42 or len(c["records"]) != len(set(c["records"])):
        raise ValueError("requires seed42 and unique records")
    seeds = c["sampling_seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("requires explicit unique nonnegative sampling seeds")
    for key in (
        "control_fps",
        "online_max_steps",
        "online_chunk_size",
        "online_execution_horizon",
        "max_observation_interval_steps",
        "stable_success_last_steps",
    ):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if (
        c["online_chunk_size"] % 4
        or c["online_chunk_size"] / c["control_fps"] > 2
        or c["online_execution_horizon"] > c["online_chunk_size"]
        or c["stable_success_last_steps"] > c["online_max_steps"]
    ):
        raise ValueError("invalid generated horizon, execution prefix or stability tail")
    position_variant(c, variant_name, source=source)
    ck = None
    if source == "wam":
        matches = [x for x in c["checkpoints"] if x["name"] == checkpoint]
        if len(matches) != 1:
            raise ValueError("requires one explicitly selected checkpoint")
        ck = matches[0]
        if type(ck["gpu"]) is not int or ck["gpu"] not in (4, 5, 6, 7):
            raise ValueError("only physical GPU4–7 may supply models")
        if type(ck["port"]) is not int or not 1024 <= ck["port"] <= 65535:
            raise ValueError("requires explicit unprivileged service port")
    elif checkpoint is not None:
        raise ValueError("oracle qualification cannot be labeled as WAM checkpoint")
    if sha256(c["fixture"]) != c["fixture_sha256"]:
        raise ValueError("fixture changed")
    fixture = json.loads(Path(c["fixture"]).read_text())
    records = {r["name"]: r for r in fixture["records"]}
    if len(records) != len(fixture["records"]) or not set(c["records"]) <= set(records):
        raise ValueError("fixture has missing or duplicate record identities")
    selected = [records[name] for name in c["records"]]
    return c, root, selected, ck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--source", choices=("wam", "oracle"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--job-indices")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", help=argparse.SUPPRESS)
    parser.add_argument("--sampling-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-snapshot", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO:
        raise ValueError("run commands from the S2A repository")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("coordinator and physics require empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    c, root, records, ck = protocol(
        args.config, source=args.source, variant_name=args.variant, checkpoint=args.checkpoint
    )
    if args.seed != c["seed"] or not 1 <= args.workers <= 4:
        raise ValueError("seed mismatch or invalid CPU worker count")
    if args.source == "wam" and args.workers != 1:
        raise ValueError("one physical worker per WAM service is mandatory")
    from s2a.stability_position_simulation import execution_inputs, run_position_case, verify_inputs

    if args.record is not None:
        if (
            args.job_indices
            or args.dry_run
            or not args.worker_snapshot
            or args.sampling_seed not in c["sampling_seeds"]
        ):
            raise ValueError("worker requires its coordinator snapshot and registered sampling seed")
        snapshot_path = Path(args.worker_snapshot).resolve()
        if not snapshot_path.is_relative_to(root):
            raise ValueError("worker snapshot must belong to this experiment")
        snapshot = json.loads(snapshot_path.read_text())
        expected = (args.source, args.variant, args.checkpoint)
        if tuple(snapshot["identity"]) != expected or snapshot["config_sha256"] != sha256(args.config):
            raise ValueError("worker snapshot identifies another protocol")
        record = next(r for r in records if r["name"] == args.record)
        if [record["name"], args.sampling_seed] not in snapshot["jobs"]:
            raise ValueError("worker job was not registered by coordinator")
        result = run_position_case(
            c,
            root,
            record,
            source=args.source,
            variant_name=args.variant,
            sampling_seed=args.sampling_seed,
            checkpoint=args.checkpoint,
            config_path=args.config,
            frozen_inputs=snapshot["execution_sha256"],
        )
        print(json.dumps(result), flush=True)
        return
    if args.worker_snapshot or args.sampling_seed is not None:
        raise ValueError("worker-only arguments require --record")
    jobs = [(r, seed) for r in records for seed in c["sampling_seeds"]]
    if args.job_indices is not None:
        indices = [int(index) for index in args.job_indices.split(",")]
        if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(jobs) for i in indices):
            raise ValueError("invalid job selection")
        jobs = [jobs[i] for i in indices]
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    jobs=[(r["name"], s) for r, s in jobs],
                    source=args.source,
                    variant=args.variant,
                    checkpoint=args.checkpoint,
                    model_started=False,
                    physics_started=False,
                )
            )
        )
        return
    label = f"{args.checkpoint or args.source}_{args.variant}"
    suffix = f"_jobs{args.job_indices.replace(',', '-')}" if args.job_indices is not None else ""
    logs = root / "simulation/logs" / (label + suffix)
    logs.mkdir(parents=True, exist_ok=False)
    for record, seed in jobs:
        case = root / "simulation" / label / f"{record['name']}_s{seed}"
        if case.exists():
            raise FileExistsError(f"refusing to rerun/overwrite a case: {case}")
    snapshot = dict(
        identity=[args.source, args.variant, args.checkpoint],
        config_sha256=sha256(args.config),
        execution_sha256=execution_inputs(c, args.config),
        jobs=[(r["name"], s) for r, s in jobs],
        coordinator_pid=os.getpid(),
        model_started=False,
        cpu_workers=args.workers,
        actual_module_scope="loaded S2A .py modules; worker supplements loaded simulator sources after fresh restore",
    )
    snapshot_path = logs / "execution_snapshot.json"
    write_json(snapshot_path, snapshot)
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
        PYTHONPATH=str(REPO / "src"),
    )

    def execute(job):
        verify_inputs(snapshot["execution_sha256"])
        record, seed = job
        command = [
            str(Path(c["simulation_python"]).absolute()),
            str(Path(__file__).absolute()),
            "--config",
            str(Path(args.config).absolute()),
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
            "--worker-snapshot",
            str(snapshot_path),
        ]
        if args.checkpoint:
            command += ["--checkpoint", args.checkpoint]
        log = logs / f"{record['name']}_s{seed}.log"
        with log.open("x") as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=REPO)
            write_json(log.with_suffix(".process.json"), dict(pid=process.pid, command=command))
            code = process.wait()
        verify_inputs(snapshot["execution_sha256"])
        result = dict(record=record["name"], sampling_seed=seed, returncode=code, command=command, pid=process.pid)
        print(json.dumps(result), flush=True)
        return result

    lock_context = nullcontext()
    if args.source == "wam":
        lock_dir = REPO / "outputs/wam_stability_20261021/service_physics_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_context = (lock_dir / f"port{ck['port']}.lock").open("a")
    with lock_context as lock:
        if args.source == "wam":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(execute, jobs))
    matrix = root / "simulation" / f"position_matrix_{label}{suffix}.json"
    if matrix.exists():
        raise FileExistsError(matrix)
    write_json(
        matrix,
        dict(
            complete=all(r["returncode"] == 0 for r in results),
            cases=results,
            execution_snapshot_sha256=sha256(snapshot_path),
            model_started=False,
        ),
    )
    if any(r["returncode"] for r in results):
        raise SystemExit("position matrix failed; original evidence retained")


if __name__ == "__main__":
    main()
