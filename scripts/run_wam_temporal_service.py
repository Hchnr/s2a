"""Own one020 model/worker only after fixed qualification and019 final release."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests
from run_wam_long_service import weight_stats
from run_wam_short_replan import cpu_environment, interrupted, process_record, stop_owned, verify_hashes, write_new

from s2a.artifacts import REPO, sha256

REGISTRATION = "7025c45"
MATRIX = "scripts/run_wam_temporal_matrix.py"
SERVER = "scripts/serve_wam_acceptance.py"


def preflight(gate_path, gate_sha, part):
    from run_wam_temporal_matrix import protocol

    assert part in (0, 1, 2)
    path = Path(gate_path).resolve()
    assert path.is_relative_to(REPO / "outputs") and sha256(path) == gate_sha
    gate = json.loads(path.read_text())
    assert gate["schema_version"] == "s2a.temporal-equal-launch-gate.v1" and gate["all_prerequisites_passed"]
    relative = f"configs/wam_temporal_equal_task{part}_20261021.json"
    original = subprocess.check_output(["git", "show", f"{REGISTRATION}:{relative}"], cwd=REPO)
    assert sha256(REPO / relative) == hashlib.sha256(original).hexdigest()
    plan = "plans/v0.0.1/020_causal_temporal_ensemble_20261021.md"
    plan_original = subprocess.check_output(["git", "show", f"{REGISTRATION}:{plan}"], cwd=REPO)
    assert sha256(REPO / plan) == hashlib.sha256(plan_original).hexdigest()
    config, root, records, ck = protocol(
        relative, source="wam", variant_name="temporal_equal", checkpoint="8400_masked"
    )
    assert config["records"] == [f"task{part}_init0"] and config["sampling_seeds"] == [42]
    assert ck["gpu"] == 4 + part and ck["port"] == 9844 + part and len(records) == 1
    signatures = dict(gate["input_sha256"])
    for name in (
        relative,
        "plans/v0.0.1/020_causal_temporal_ensemble_20261021.md",
        MATRIX,
        SERVER,
        "src/s2a/temporal_ensemble.py",
        "src/s2a/temporal_ensemble_simulation.py",
        "src/s2a/wam_runtime.py",
        "scripts/run_wam_temporal_service.py",
        "scripts/capture_wam_temporal_ready.py",
    ):
        assert signatures[str(REPO / name)] == sha256(REPO / name)
    for label in ("qualification", "prior_seed43"):
        source = gate[label]
        assert signatures[source["path"]] == source["sha256"] == sha256(source["path"])
        item = json.loads(Path(source["path"]).read_text())
        assert item["complete"] and item["normal_process_release"]
        if label == "qualification":
            assert item["qualified"] and item["verified_oracle_cases"] == 3
            assert item["verified_physics_log_endpoints"] == 1800 and item["all1800_common_step_fields_exact_to_007"]
        else:
            assert item["schema_version"] == "s2a.raw8-seed43-independent-audit.v1"
            assert item["all_cases_independently_verified"] and item["verified_cases"] == 3
            assert item["registered_budget"] == dict(
                wam_trials=3, physics_steps=1800, h40_queries=225, new_oracle_trials=0, fresh_initial_states=0
            )
            assert item["verified_physics_steps"] == 1800 and item["verified_queries"] == 225
            assert item["verified_http_queries"] == 225 and item["errors"] == []
        for name, digest in item["input_sha256"].items():
            assert signatures[name] == digest
    assert str(path) not in signatures or signatures[str(path)] == gate_sha
    signatures[str(path)] = gate_sha
    verify_hashes(signatures)
    stats = weight_stats(ck)
    model_command = [
        str(Path(config["cosmos_repo"]) / ".venv/bin/python"),
        SERVER,
        "--config",
        relative,
        "--seed",
        "42",
        "--checkpoint",
        "8400_masked",
        "--port",
        str(ck["port"]),
        "--active-slot-mask",
    ]
    job_command = [
        str(REPO / ".venv/bin/python"),
        MATRIX,
        "--config",
        relative,
        "--seed",
        "42",
        "--source",
        "wam",
        "--variant",
        "temporal_equal",
        "--checkpoint",
        "8400_masked",
        "--workers",
        "1",
    ]
    return config, root, ck, signatures, stats, model_command, job_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cpu_environment()
    assert Path.cwd().resolve() == REPO
    config, root, ck, signatures, stats, model_command, job_command = preflight(
        args.gate, args.gate_sha256, args.partition
    )
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    physical_gpu=ck["gpu"],
                    port=ck["port"],
                    model_command=model_command,
                    job_command=job_command,
                    actual_checkpoint_stat=stats,
                    new_model_requests=0,
                    new_physics_steps=0,
                )
            )
        )
        return
    directory = root / "orchestration"
    directory.mkdir(parents=True, exist_ok=False)
    lock_path = REPO / f"outputs/wam_stability_20261021/gpu_locks/physical{ck['gpu']}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    model = job = None
    model_identity = job_identity = None
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = subprocess.check_output(
            ["nvidia-smi", f"--id={ck['gpu']}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        if int(memory.strip()) != 0:
            raise RuntimeError("020 requires actual allowed-card memory strictly0; no existing process is stopped")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", ck["port"]))
        manifest = dict(
            schema_version="s2a.temporal-equal-service.v1",
            partition=args.partition,
            physical_gpu=ck["gpu"],
            coordinator=process_record(os.getpid(), ""),
            config=config,
            config_sha256=sha256(f"configs/wam_temporal_equal_task{args.partition}_20261021.json"),
            source_sha256=signatures,
            model_command=model_command,
            job_command=job_command,
            actual_checkpoint_stat=stats,
            automatic_retry=False,
            external_sources_readonly=True,
        )
        write_new(directory / "manifest.json", manifest)
        model_environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(ck["gpu"]),
            LD_LIBRARY_PATH="",
            MASTER_PORT=str(30900 + ck["gpu"]),
            COSMOS_TRAINING="0",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=config["cosmos_repo"] + ":" + str(REPO / "src"),
            OMP_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false",
            TORCHINDUCTOR_CACHE_DIR=str(root / "model_cache/inductor"),
            TRITON_CACHE_DIR=str(root / "model_cache/triton"),
        )
        try:
            verify_hashes(signatures)
            with (directory / "service.log").open("x") as stream:
                model = subprocess.Popen(
                    model_command,
                    cwd=REPO,
                    env=model_environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            model_identity = process_record(model.pid, str(ck["gpu"]))
            assert model_identity["present"]
            write_new(directory / "model_process.json", dict(model_identity, command=model_command))
            with requests.Session() as session:
                session.trust_env = False
                for _ in range(180):
                    if model.poll() is not None:
                        raise RuntimeError("owned020 model exited before readiness")
                    try:
                        response = session.get(f"http://127.0.0.1:{ck['port']}/info", timeout=2)
                        response.raise_for_status()
                        info = response.json()
                        break
                    except requests.RequestException:
                        time.sleep(5)
                else:
                    raise TimeoutError("owned020 model readiness timed out")
            assert Path(info["checkpoint_path"]) == Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
            assert (
                not info["ema_weights"]
                and info["acceptance_intervention"]["active_slot_mask"]
                and info["attention_backend"] == "sdpa"
            )
            write_new(directory / "server_info.json", info)
            verify_hashes(signatures)
            with (directory / "matrix.log").open("x") as stream:
                job = subprocess.Popen(
                    job_command,
                    cwd=REPO,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                job_identity = process_record(job.pid, "")
                assert job_identity["present"]
                write_new(directory / "matrix_process.json", dict(job_identity, command=job_command))
                if job.wait():
                    raise RuntimeError("020 physics matrix failed; no task-performance conclusion inferred")
            verify_hashes(signatures)
            weight_stats(ck)
            case = root / "simulation/8400_masked_temporal_equal" / f"task{args.partition}_init0_s42"
            result = json.loads((case / "result.json").read_text())
            assert result["complete"] and result["steps"] == 600 and result["chunks"] == 75
            write_new(
                directory / "complete.json",
                dict(complete=True, partition=args.partition, wam_trials=1, physics_steps=600, queries=75),
            )
        except BaseException as exc:
            write_new(
                directory / "error.json",
                dict(error=repr(exc), scientific_failure_inferred=False, automatic_retry=False),
            )
            raise
        finally:
            cleanup, errors = {}, []
            for label, process, identity, cuda in (
                ("job", job, job_identity, ""),
                ("model", model, model_identity, str(ck["gpu"])),
            ):
                try:
                    _, cleanup[label] = stop_owned(process, identity, cuda)
                except BaseException as exc:
                    errors.append(dict(process=label, error=repr(exc)))
            write_new(
                directory / "release.json",
                dict(
                    model_pid=model.pid if model else None,
                    model_start_ticks=model_identity["start_ticks"] if model_identity else None,
                    model_returncode=model.poll() if model else None,
                    last_job_pid=job.pid if job else None,
                    last_job_returncode=job.poll() if job else None,
                    stop_evidence=cleanup,
                    cleanup_errors=errors,
                    only_owned_process_groups_stopped=True,
                    automatic_retry=False,
                ),
            )
            if errors:
                raise RuntimeError("owned020 cleanup failed; inspect retained release evidence")


if __name__ == "__main__":
    main()
