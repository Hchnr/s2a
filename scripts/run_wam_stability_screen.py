"""Own one allowed GPU service and a CPU diagnostic state screen, then release it."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import requests

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def stop_child(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def interrupted(signum, _frame):
    raise InterruptedError(f"owned screen interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    parser.add_argument("--fingerprints", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"] or len(c["checkpoints"]) != 1:
        raise ValueError("requires one explicitly configured checkpoint and seed42")
    ck = c["checkpoints"][0]
    gpu, port = ck["gpu"], ck["port"]
    if gpu not in (4, 5, 6, 7) or not 1024 <= port <= 65535:
        raise ValueError("only physical GPU4–7 and an explicit unprivileged port are allowed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("the CPU coordinator requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    if c["offline_horizons"] != [40] or c["offline_prompt_modes"] != ["task_only"]:
        raise ValueError("this screen requires H40 and task-only prompts")
    samples = json.loads((root / "samples/manifest.json").read_text())
    if len(samples["cases"]) != 24 or any(s["split"] != "diagnostic" for s in samples["cases"]):
        raise ValueError("requires the frozen24 diagnostic anchors; no heldout selection")
    fingerprints = json.loads(Path(args.fingerprints).read_text())
    weights = (
        next(x for x in fingerprints["checkpoints"] if x["name"] == ck["name"])
        if "checkpoints" in fingerprints
        else fingerprints
    )
    expected_path = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
    if Path(weights.get("checkpoint", weights.get("checkpoint_path"))).resolve() != expected_path.resolve():
        raise ValueError("complete fingerprint belongs to a different checkpoint")

    def check_weight_stats():
        for filename, signature in weights["files"].items():
            stat = Path(filename).stat()
            if stat.st_size != signature["bytes"] or stat.st_mtime_ns != signature["mtime_ns"]:
                raise ValueError(f"weight changed since complete byte hashing: {filename}")

    check_weight_stats()
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
    job_cmd = [
        str(Path(".venv/bin/python").absolute()),
        "scripts/run_wam_state_matrix.py",
        "--config",
        args.config,
        "--seed",
        str(args.seed),
        "--checkpoint",
        ck["name"],
    ]
    if args.dry_run:
        print(json.dumps(dict(gpu=gpu, port=port, queries=48, model=model_cmd, job=job_cmd)))
        return
    dest = root / "orchestration"
    dest.mkdir(exist_ok=False)
    lock_path = Path("outputs/wam_stability_20261021/gpu_locks")
    lock_path.mkdir(parents=True, exist_ok=True)
    model, job = None, None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    with (lock_path / f"physical{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        if int(memory.strip()) >= 500:
            raise RuntimeError("allowed GPU is occupied; no other process was stopped")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
        write_json(
            dest / "manifest.json",
            dict(
                config=c,
                code=code_fingerprint(),
                coordinator_pid=os.getpid(),
                config_sha256=sha256(args.config),
                samples_sha256=sha256(root / "samples/manifest.json"),
                checkpoint_fingerprints_sha256=sha256(args.fingerprints),
                physical_gpu=gpu,
                model_command=model_cmd,
                job_command=job_cmd,
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
            with (dest / "service.log").open("x") as model_log, (dest / "screen.log").open("x") as job_log:
                model = subprocess.Popen(
                    model_cmd, env=model_env, stdout=model_log, stderr=subprocess.STDOUT, start_new_session=True
                )
                write_json(
                    dest / "model_process.json",
                    dict(
                        pid=model.pid,
                        gpu=gpu,
                        port=port,
                        command=model_cmd,
                        cuda_visible_devices=str(gpu),
                        coordinator_pid=os.getpid(),
                    ),
                )
                with requests.Session() as session:
                    session.trust_env = False
                    endpoint = f"http://127.0.0.1:{port}"
                    for _ in range(180):
                        if model.poll() is not None:
                            raise RuntimeError("owned model exited before readiness; inspect service.log")
                        try:
                            response = session.get(endpoint + "/info", timeout=2)
                            response.raise_for_status()
                            info = response.json()
                            break
                        except requests.RequestException:
                            time.sleep(5)
                    else:
                        raise TimeoutError("owned model readiness timeout")
                    expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
                    if Path(info["checkpoint_path"]).resolve() != expected.resolve() or info["ema_weights"]:
                        raise ValueError("service checkpoint identity differs")
                    if not info["acceptance_intervention"]["active_slot_mask"]:
                        raise ValueError("active-slot mask intervention missing")
                    write_json(dest / "server_info.json", info)
                    if ck.get("regression_reference"):
                        reference = Path(ck["regression_reference"])
                        request = json.loads((reference / "request.json").read_text())
                        request["name"] = "stability_regression_before_screen"
                        response = session.post(endpoint + "/generate", json=request, timeout=900)
                        response.raise_for_status()
                        raw = response.json()
                        write_json(dest / "regression_response.json", raw)
                        actual = np.asarray(raw["outputs"][0]["content"]["action"])
                        old = np.asarray(
                            json.loads((reference / "response.json").read_text())["outputs"][0]["content"]["action"]
                        )
                        identical = bool(np.array_equal(actual, old))
                        write_json(
                            dest / "regression_check.json",
                            dict(
                                identical=identical,
                                max_abs=float(np.max(np.abs(actual - old))),
                                request_source_sha256=sha256(reference / "request.json"),
                                response_source_sha256=sha256(reference / "response.json"),
                            ),
                        )
                        if not identical:
                            raise ValueError("old8800 service no longer reproduces its fixed regression")
                job = subprocess.Popen(
                    job_cmd,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                    stdout=job_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                write_json(dest / "screen_process.json", dict(pid=job.pid, command=job_cmd))
                if job.wait():
                    raise RuntimeError("state diagnostic screen failed; original results retained")
                complete = root / "state_queries" / ck["name"] / "complete.json"
                result = json.loads(complete.read_text())
                if result["cases"] != 48:
                    raise ValueError("screen did not complete its full48 query budget")
                check_weight_stats()
                write_json(dest / "complete.json", dict(complete=True, queries=48, screen_sha256=sha256(complete)))
        except BaseException as exc:
            write_json(dest / "error.json", dict(error=repr(exc), scientific_failure_inferred=False))
            raise
        finally:
            stop_child(job)
            stop_child(model)
            write_json(
                dest / "release.json",
                dict(
                    model_pid=model.pid if model else None,
                    model_returncode=model.poll() if model else None,
                    job_pid=job.pid if job else None,
                    job_returncode=job.poll() if job else None,
                ),
            )


if __name__ == "__main__":
    main()
