"""Serve a frozen Cosmos checkpoint from S2A, after waiting for an unused GPU."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from s2a.artifacts import load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_simulation.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int)
    parser.add_argument("--max-wait-seconds", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, output = load_config(args.config)
    port = args.port if args.port is not None else 8180 + args.gpu
    serving_dir = output / ("serving" if port == 8180 else f"serving_{port}")
    suffix = "" if port == 8180 else f"_{port}"
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen experiment")
    repo = Path(config["cosmos_repo"])
    run = Path(config["checkpoint_run"])
    iteration = run / "checkpoints" / config["checkpoint_iteration"]
    if not (iteration / ".save_complete").is_file():
        raise ValueError("frozen checkpoint is not marked complete")
    command = [str(repo / ".venv/bin/python")]
    if not args.dry_run:
        command += [
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=1",
            "--master-addr=127.0.0.1",
            f"--master-port={29880 + args.gpu}",
        ]
    command += [
        "-m",
        "cosmos_framework.scripts.action128_server",
        "--checkpoint-path",
        str(iteration / "model"),
        "--output-dir",
        str(serving_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--attention-backend",
        "sdpa",
        "--no-compile",
        "--no-reuse-small-action-metadata",
    ]
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": "",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "COSMOS_TRAINING": "0",
        "PYTHONPATH": str(repo),
        "OMP_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(output / "inductor"),
    }
    if args.dry_run:
        raise SystemExit(subprocess.call(command + ["--dry-run"], env=env))
    started, idle_since = time.monotonic(), None
    while True:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
        memory, utilization = map(int, rows[args.gpu].split(","))
        idle = memory < 500 and utilization == 0
        idle_since = (idle_since or time.monotonic()) if idle else None
        waiting = dict(stage="waiting_for_unused_gpu", gpu=args.gpu, memory_used_mib=memory, utilization=utilization)
        write_json(output / f"server_status{suffix}.json", waiting)
        if idle_since is not None and time.monotonic() - idle_since >= 10:
            break
        if time.monotonic() - started > args.max_wait_seconds:
            raise TimeoutError("No unused GPU became available; other jobs were left running")
        time.sleep(10)
    snapshot = {
        "command": command,
        "config_sha256": sha256(run / "config.yaml"),
        "metadata_sha256": sha256(iteration / "model/.metadata"),
        "completion_sha256": sha256(iteration / ".save_complete"),
        "checkpoint_files": {p.name: p.stat().st_size for p in sorted((iteration / "model").iterdir()) if p.is_file()},
        "cosmos_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "gpu": args.gpu,
    }
    write_json(output / f"server_launch{suffix}.json", snapshot)
    with (output / f"serving{suffix}.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / f"server_process{suffix}.json", {**snapshot, "pid": process.pid})
        write_json(output / f"server_status{suffix}.json", dict(stage="loading", pid=process.pid, gpu=args.gpu))
        code = process.wait()
    write_json(output / f"server_status{suffix}.json", dict(stage="exited", pid=process.pid, exit_code=code))
    print(json.dumps(dict(exit_code=code)), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
