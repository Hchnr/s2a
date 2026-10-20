"""Own one allowed model replica and queued causal WAM physics matrices."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests
from run_wam_stability_screen import interrupted, stop_child

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variants", required=True, help="comma-separated, pre-registered variant names")
    parser.add_argument("--fingerprints", required=True)
    parser.add_argument("--job-indices")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    c, root = load_config(args.config)
    ck = next(ck for ck in c["checkpoints"] if ck["name"] == args.checkpoint)
    variants = args.variants.split(",")
    if args.seed != c["seed"] or len(variants) != len(set(variants)):
        raise ValueError("seed mismatch or repeated variant")
    if not set(variants) <= {v["name"] for v in c["variants"]}:
        raise ValueError("unknown executor variant")
    gpu, port = ck["gpu"], ck["port"]
    if gpu not in (4, 5, 6, 7) or not 1024 <= port <= 65535:
        raise ValueError("only physical GPU4–7 with explicit unprivileged ports")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU coordinator requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    fingerprints = json.loads(Path(args.fingerprints).read_text())
    weights = (
        next(x for x in fingerprints["checkpoints"] if x["name"] == ck["name"])
        if "checkpoints" in fingerprints
        else fingerprints
    )
    expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
    if Path(weights.get("checkpoint", weights.get("checkpoint_path"))).resolve() != expected.resolve():
        raise ValueError("fingerprint identifies another checkpoint")

    def check_weights():
        for name, item in weights["files"].items():
            stat = Path(name).stat()
            if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError(f"checkpoint file changed since byte hashing: {name}")

    check_weights()
    model_cmd = [
        str(Path(c["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_acceptance.py",
        "--config",
        args.config,
        "--seed",
        str(args.seed),
        "--checkpoint",
        ck["name"],
        "--port",
        str(port),
        "--active-slot-mask",
    ]
    jobs = []
    for variant in variants:
        cmd = [
            str(Path(".venv/bin/python").absolute()),
            "scripts/run_wam_stability_matrix.py",
            "--config",
            args.config,
            "--seed",
            str(args.seed),
            "--source",
            "wam",
            "--variant",
            variant,
            "--checkpoint",
            ck["name"],
        ]
        if args.job_indices:
            cmd += ["--job-indices", args.job_indices]
        jobs.append((variant, cmd))
    if args.dry_run:
        print(json.dumps(dict(physical_gpu=gpu, model=model_cmd, jobs=jobs)))
        return
    suffix = "_jobs" + args.job_indices.replace(",", "-") if args.job_indices else ""
    dest = root / "orchestration" / (ck["name"] + "_" + "_".join(variants) + suffix)
    dest.mkdir(parents=True, exist_ok=False)
    lock_path = Path("outputs/wam_stability_20261021/gpu_locks")
    lock_path.mkdir(parents=True, exist_ok=True)
    model, job = None, None
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    with (lock_path / f"physical{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        if int(memory.strip()) >= 500:
            raise RuntimeError("allowed GPU occupied; no existing process was stopped")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
        write_json(
            dest / "manifest.json",
            dict(
                config=c,
                config_sha256=sha256(args.config),
                fingerprint_manifest_sha256=sha256(args.fingerprints),
                code=code_fingerprint(),
                coordinator_pid=os.getpid(),
                physical_gpu=gpu,
                model_command=model_cmd,
                jobs=jobs,
            ),
        )
        model_env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            LD_LIBRARY_PATH="",
            MASTER_PORT=str(30900 + gpu),
            COSMOS_TRAINING="0",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=c["cosmos_repo"] + ":" + str(Path("src").absolute()),
            OMP_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false",
            TORCHINDUCTOR_CACHE_DIR=str(root / "model_cache"),
        )
        try:
            with (dest / "service.log").open("x") as stream:
                model = subprocess.Popen(
                    model_cmd, env=model_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
                write_json(dest / "model_process.json", dict(pid=model.pid, gpu=gpu, port=port))
                with requests.Session() as session:
                    session.trust_env = False
                    for _ in range(180):
                        if model.poll() is not None:
                            raise RuntimeError("owned model exited before readiness")
                        try:
                            response = session.get(f"http://127.0.0.1:{port}/info", timeout=2)
                            response.raise_for_status()
                            info = response.json()
                            break
                        except requests.RequestException:
                            time.sleep(5)
                    else:
                        raise TimeoutError("owned model readiness timeout")
                if (
                    Path(info["checkpoint_path"]).resolve() != expected.resolve()
                    or info["ema_weights"]
                    or not info["acceptance_intervention"]["active_slot_mask"]
                ):
                    raise ValueError("wrong model or inactive mask absent")
                write_json(dest / "server_info.json", info)
                for variant, command in jobs:
                    with (dest / f"{variant}.log").open("x") as log:
                        job = subprocess.Popen(
                            command,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        write_json(dest / f"{variant}_process.json", dict(pid=job.pid, command=command))
                        if job.wait():
                            raise RuntimeError(f"physics matrix failed: {variant}; no scientific conclusion inferred")
                    check_weights()
                write_json(dest / "complete.json", dict(complete=True, variants=variants))
        except BaseException as exc:
            write_json(dest / "error.json", dict(error=repr(exc), scientific_failure_inferred=False))
            raise
        finally:
            stop_child(job)
            stop_child(model)
            write_json(
                dest / "release.json",
                dict(model_pid=model.pid if model else None, model_returncode=model.poll() if model else None),
            )


if __name__ == "__main__":
    main()
