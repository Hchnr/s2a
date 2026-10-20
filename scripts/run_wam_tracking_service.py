"""Own the registered016 model and one causal, raw-reference physics matrix."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from run_wam_short_replan import cpu_environment, interrupted, process_record, stop_owned, verify_hashes, write_new
from run_wam_tracking_matrix import protocol

from s2a.artifacts import REPO, sha256

CONFIG = "configs/tracking_replan_wam_20261021.json"
MATRIX = "scripts/run_wam_tracking_matrix.py"
SERVER = "scripts/serve_wam_acceptance.py"
FINGERPRINT = "outputs/wam_independent_20261020/checkpoint_fingerprints.json"
FINGERPRINT_SHA = "624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8"


def weight_stats(checkpoint):
    if sha256(REPO / FINGERPRINT) != FINGERPRINT_SHA:
        raise ValueError("original complete checkpoint fingerprint differs")
    weights = next(
        x for x in json.loads((REPO / FINGERPRINT).read_text())["checkpoints"] if x["name"] == checkpoint["name"]
    )
    expected = Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model"
    if Path(weights["checkpoint"]).resolve() != expected.resolve() or len(weights["files"]) != 163:
        raise ValueError("original163-file fingerprint identifies another checkpoint")
    for filename, item in weights["files"].items():
        stat = Path(filename).stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"] or len(item["sha256"]) != 64:
            raise ValueError(f"checkpoint changed since full byte hashing: {filename}")
    return dict(files_stat_rechecked=163, original_full_fingerprint_sha256=FINGERPRINT_SHA, weight_rehashed=False)


def preflight(path, digest):
    path = Path(path).resolve()
    if not path.is_relative_to(REPO / "outputs") or sha256(path) != digest:
        raise ValueError("launch gate must be the explicit unchanged repository artifact")
    gate = json.loads(path.read_text())
    if gate["schema_version"] != "s2a.tracking-replan-launch-gate.v1" or not gate["all_prerequisites_passed"]:
        raise ValueError("016 launch prerequisites are incomplete")
    c, root, records, ck = protocol(CONFIG, source="wam", variant_name="position_hold", checkpoint="8400_masked")
    if (
        ck["gpu"] != 7
        or ck["port"] != 9597
        or c["records"] != ["task0_init0", "task1_init0", "task2_init0"]
        or c["sampling_seeds"] != [42]
        or (c["online_max_steps"], c["sampling_steps"], c["guidance"], c["shift"]) != (600, 35, 1, 5)
        or c["control_fps"] != 20
        or c["stable_success_last_steps"] != 40
    ):
        raise ValueError("016 fixed resource/old-scene/three-trial identity differs")
    signatures = dict(gate["input_sha256"])
    if signatures.get(str(REPO / CONFIG)) != sha256(REPO / CONFIG):
        raise ValueError("launch gate omitted actual configuration")
    for relative in (
        "plans/v0.0.1/016_tracking_replan_20261021.md",
        "src/s2a/tracking_replan.py",
        "src/s2a/stability_tracking_simulation.py",
        MATRIX,
        SERVER,
        "src/s2a/wam_runtime.py",
        "scripts/run_wam_short_replan.py",
        "scripts/run_wam_tracking_service.py",
    ):
        if signatures.get(str(REPO / relative)) != sha256(REPO / relative):
            raise ValueError(f"launch gate omitted or changed actual dependency: {relative}")
    signatures[str(path)] = digest
    verify_hashes(signatures)
    qualification = json.loads(Path(gate["qualification"]["path"]).read_text())
    if (
        sha256(gate["qualification"]["path"]) != gate["qualification"]["sha256"]
        or not qualification["qualification_passed"]
        or not qualification["all_cases_complete"]
        or qualification["oracle_trials"] != 3
        or qualification["physical_steps"] != 1800
    ):
        raise ValueError("016 three-case independent execution qualification is incomplete")
    for name, digest_value in qualification["input_sha256"].items():
        if signatures.get(name) != digest_value:
            raise ValueError("launch gate omitted a qualification input")
    backend = gate["backend_release"]
    if sha256(backend["path"]) != backend["sha256"] or signatures.get(backend["path"]) != backend["sha256"]:
        raise ValueError("013 canonical quality/release artifact is not source-bound")
    checked_backend = json.loads(Path(backend["path"]).read_text())
    if (
        not checked_backend["quality_complete"]
        or checked_backend["verified_queries"] != 48
        or not checked_backend["owned_service_released_normally"]
        or checked_backend["errors"]
        or checked_backend["release"]["model_returncode"] != -15
        or checked_backend["release"]["job_returncode"] != 0
        or checked_backend["release"]["cleanup_errors"]
    ):
        raise ValueError("013 quality budget or controlled release incomplete")
    for name, digest_value in checked_backend["input_sha256"].items():
        if signatures.get(name) != digest_value:
            raise ValueError("launch gate omitted a013 audit input")
    for owned in backend["owned_processes"]:
        if process_record(owned["pid"], owned["cuda_visible_devices"], start_ticks=owned["start_ticks"])["present"]:
            raise RuntimeError("013 owned process is still live")
    return c, root, records, ck, signatures, weight_stats(ck)


def commands(c, ck):
    model = [
        str(Path(c["cosmos_repo"]) / ".venv/bin/python"),
        SERVER,
        "--config",
        CONFIG,
        "--seed",
        "42",
        "--checkpoint",
        ck["name"],
        "--port",
        str(ck["port"]),
        "--active-slot-mask",
    ]
    job = [
        str(REPO / ".venv/bin/python"),
        MATRIX,
        "--config",
        CONFIG,
        "--seed",
        "42",
        "--source",
        "wam",
        "--variant",
        "position_hold",
        "--checkpoint",
        ck["name"],
    ]
    return model, job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cpu_environment()
    c, root, records, ck, signatures, stats = preflight(args.gate, args.gate_sha256)
    model_command, job_command = commands(c, ck)
    summary = dict(
        schema_version="s2a.tracking-replan-coordinator.v1",
        physical_gpu=7,
        port=9597,
        wam_trials=3,
        physics_steps=1800,
        maximum_h40_queries=225,
        model_command=model_command,
        job_command=job_command,
        weight_stats=stats,
        source_sha256=signatures,
    )
    if args.dry_run:
        print(json.dumps(dict(summary, model_started=False, physics_started=False)))
        return
    if (root / "simulation").exists() or (root / "intervention_services").exists():
        raise FileExistsError("refusing to repeat/overwrite016 model or physics evidence")
    directory = root / "orchestration"
    directory.mkdir(parents=True, exist_ok=False)
    locks = REPO / "outputs/wam_stability_20261021/gpu_locks"
    locks.mkdir(parents=True, exist_ok=True)
    model = job = None
    model_identity = job_identity = None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    with (locks / "physical7.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            preflight(args.gate, args.gate_sha256)
            memory = int(
                subprocess.check_output(
                    ["nvidia-smi", "--id=7", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
                ).strip()
            )
            if memory != 0:
                raise RuntimeError("allowed GPU7 occupied; no other process stopped")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 9597))
            write_new(
                directory / "manifest.json",
                dict(
                    summary,
                    coordinator=process_record(os.getpid(), ""),
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    gpu_memory_mib_before=memory,
                ),
            )
            model_env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="7",
                LD_LIBRARY_PATH="",
                MASTER_PORT="30907",
                COSMOS_TRAINING="0",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=c["cosmos_repo"] + ":" + str(REPO / "src"),
                OMP_NUM_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
                TORCHINDUCTOR_CACHE_DIR=str(root / "model_cache"),
            )
            with (directory / "service.log").open("x") as stream:
                model = subprocess.Popen(
                    model_command,
                    env=model_env,
                    cwd=REPO,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                model_identity = process_record(model.pid, "7")
                if not model_identity["present"]:
                    raise RuntimeError("owned model exited before registration")
                write_new(directory / "model_process.json", dict(model_identity, command=model_command))
                with requests.Session() as session:
                    session.trust_env = False
                    for _ in range(180):
                        if model.poll() is not None:
                            raise RuntimeError("owned model exited before readiness")
                        try:
                            response = session.get("http://127.0.0.1:9597/info", timeout=2)
                            response.raise_for_status()
                            info = response.json()
                            break
                        except requests.RequestException:
                            time.sleep(5)
                    else:
                        raise TimeoutError("model readiness timed out; no automatic retry")
                expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
                intervention = info["acceptance_intervention"]
                if (
                    Path(info["checkpoint_path"]).resolve() != expected.resolve()
                    or Path(info["config_file"]).resolve() != (Path(ck["run"]) / "config.yaml").resolve()
                    or info["port"] != 9597
                    or info["attention_backend"] != "sdpa"
                    or any(info[k] != 1 for k in ("world_size", "dp_shard_size", "cp_size", "cfgp_size"))
                    or any(
                        info[k]
                        for k in (
                            "compile",
                            "fuse_qkv",
                            "cuda_graphs",
                            "reuse_small_action_und",
                            "reuse_small_action_metadata",
                            "fixed_training_buckets",
                        )
                    )
                    or info["ema_weights"]
                    or not intervention["active_slot_mask"]
                    or intervention["exact_published_caption"]
                    or intervention["warm_start_contract_run"] is not None
                ):
                    raise ValueError("owned model identity/intervention differs")
                for relative in (SERVER, "src/s2a/wam_runtime.py"):
                    if intervention["code"]["files_sha256"][relative] != signatures[str(REPO / relative)]:
                        raise ValueError("actual model service source differs")
                write_new(directory / "server_info.json", info)
                verify_hashes(signatures)
                with (directory / "matrix.log").open("x") as stream:
                    job = subprocess.Popen(
                        job_command,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
                        cwd=REPO,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    job_identity = process_record(job.pid, "")
                    if not job_identity["present"]:
                        raise RuntimeError("owned matrix exited before registration")
                    write_new(directory / "matrix_process.json", dict(job_identity, command=job_command))
                    code = job.wait()
                    write_new(directory / "matrix_exit.json", dict(pid=job.pid, returncode=code))
                    if code:
                        raise RuntimeError("matrix failed; original evidence retained without automatic retry")
                query_count = 0
                for record in records:
                    case = root / "simulation/8400_masked_position_hold" / f"{record['name']}_s42"
                    result = json.loads((case / "result.json").read_text())
                    count = len(list(case.glob("request_*.json")))
                    if (
                        not result["complete"]
                        or result["error"] is not None
                        or result["steps"] != 600
                        or not 15 <= count <= 75
                        or result["chunks"] != count
                    ):
                        raise ValueError("016 finished matrix differs from three complete bounded-query cases")
                    query_count += count
                verify_hashes(signatures)
                weight_stats(ck)
                write_new(
                    directory / "complete.json",
                    dict(complete=True, wam_trials=3, physics_steps=1800, queries=query_count),
                )
        except BaseException as exc:
            write_new(
                directory / "error.json",
                dict(error=repr(exc), scientific_failure_inferred=False, automatic_retry=False),
            )
            raise
        finally:
            cleanup_errors, codes, stops = [], {}, {}
            for name, process, identity, cuda in (
                ("job", job, job_identity, ""),
                ("model", model, model_identity, "7"),
            ):
                try:
                    codes[name], stops[name] = stop_owned(process, identity, cuda)
                except BaseException as exc:
                    cleanup_errors.append(dict(process=name, error=repr(exc)))
            write_new(
                directory / "release.json",
                dict(
                    model_pid=model.pid if model else None,
                    model_start_ticks=model_identity.get("start_ticks") if model_identity else None,
                    model_returncode=codes.get("model"),
                    last_job_pid=job.pid if job else None,
                    last_job_returncode=codes.get("job"),
                    stop_evidence=stops,
                    cleanup_errors=cleanup_errors,
                    only_owned_process_groups_stopped=True,
                    automatic_retry=False,
                ),
            )
            if cleanup_errors:
                raise RuntimeError(
                    "owned cleanup incomplete; inspect recorded identities without stopping other processes"
                )


if __name__ == "__main__":
    main()
