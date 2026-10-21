"""Own the sole021 task2 continuation on GPU6 after a source-bound zero-effect gate."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import requests
from prepare_wam_temporal_continuation import BUDGET, validate_gate
from run_wam_long_service import weight_stats
from run_wam_short_replan import cpu_environment, interrupted, process_record, stop_owned, verify_hashes, write_new
from run_wam_temporal_continuation_matrix import CONFIG, protocol

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import owned_child


def preflight(gate_path, gate_sha):
    gate = validate_gate(gate_path, gate_sha)
    config, root, _, checkpoint = protocol(CONFIG)
    signatures = dict(gate["input_sha256"])
    if str(Path(gate_path).resolve()) in signatures:
        raise ValueError("gate must not depend on its own digest")
    signatures[str(Path(gate_path).resolve())] = gate_sha
    verify_hashes(signatures)
    model_command = [
        str(Path(config["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_acceptance.py",
        "--config",
        CONFIG,
        "--seed",
        "42",
        "--checkpoint",
        "8400_masked",
        "--port",
        "9946",
        "--active-slot-mask",
    ]
    job_command = [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_temporal_continuation_matrix.py",
        "--config",
        CONFIG,
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
    return gate, config, root, checkpoint, signatures, model_command, job_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=int, choices=(2,), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cpu_environment()
    if Path.cwd().resolve() != REPO:
        raise ValueError("run continuation from repository root")
    gate, config, root, checkpoint, signatures, model_command, job_command = preflight(args.gate, args.gate_sha256)
    stats = weight_stats(checkpoint)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    partition=2,
                    physical_gpu=6,
                    port=9946,
                    registered_budget=BUDGET,
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
    lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks/physical6.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    model = job = None
    model_identity = job_identity = None
    ownership = ExitStack()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = subprocess.check_output(
            ["nvidia-smi", "--id=6", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        if int(memory.strip()) != 0:
            raise RuntimeError("continuation requires physicalGPU6 memory strictly0; existing processes are untouched")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 9946))
        manifest = dict(
            schema_version="s2a.temporal-zero-step-continuation-service.v1",
            partition=2,
            physical_gpu=6,
            coordinator=process_record(os.getpid(), ""),
            config=config,
            config_sha256=sha256(REPO / CONFIG),
            source_sha256=signatures,
            model_command=model_command,
            job_command=job_command,
            actual_checkpoint_stat=stats,
            original_evidence=gate["original_evidence"],
            source_review=gate["source_review"],
            automatic_retry=False,
            external_sources_readonly=True,
        )
        write_new(directory / "manifest.json", manifest)
        model_environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="6",
            LD_LIBRARY_PATH="",
            MASTER_PORT="30906",
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
            model_ownership = directory / "model_ownership"
            model_ownership.mkdir(exist_ok=False)
            with (directory / "service.log").open("x") as stream:
                model, _ = ownership.enter_context(
                    owned_child(
                        model_command,
                        cwd=REPO,
                        environment=model_environment,
                        folder=model_ownership,
                        stream=stream,
                        start_new_session=True,
                    )
                )
            model_identity = process_record(model.pid, "6")
            if not model_identity["present"]:
                raise RuntimeError("owned continuation model exited before identity registration")
            write_new(directory / "model_process.json", dict(model_identity, command=model_command))
            with requests.Session() as session:
                session.trust_env = False
                for _ in range(180):
                    if model.poll() is not None:
                        raise RuntimeError("owned continuation model exited before readiness")
                    try:
                        response = session.get("http://127.0.0.1:9946/info", timeout=2)
                        response.raise_for_status()
                        info = response.json()
                        break
                    except requests.RequestException:
                        time.sleep(5)
                else:
                    raise TimeoutError("owned continuation model readiness timed out")
            if (
                Path(info["checkpoint_path"])
                != Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model"
                or info["ema_weights"]
                or not info["acceptance_intervention"]["active_slot_mask"]
                or info["attention_backend"] != "sdpa"
            ):
                raise ValueError("continuation service checkpoint/backend/mask differs")
            write_new(directory / "server_info.json", info)
            verify_hashes(signatures)
            job_ownership = directory / "job_ownership"
            job_ownership.mkdir(exist_ok=False)
            with (directory / "matrix.log").open("x") as stream:
                job, _ = ownership.enter_context(
                    owned_child(
                        job_command,
                        cwd=REPO,
                        environment=dict(
                            os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"
                        ),
                        folder=job_ownership,
                        stream=stream,
                        start_new_session=True,
                    )
                )
                job_identity = process_record(job.pid, "")
                if not job_identity["present"]:
                    raise RuntimeError("owned continuation matrix exited before registration")
                write_new(directory / "matrix_process.json", dict(job_identity, command=job_command))
                if job.wait():
                    raise RuntimeError("continuation matrix failed; no task-performance failure inferred")
            verify_hashes(signatures)
            weight_stats(checkpoint)
            case = root / "simulation/8400_masked_temporal_equal/task2_init0_s42"
            result = json.loads((case / "result.json").read_text())
            if not result["complete"] or result["steps"] != 600 or result["chunks"] != 75:
                raise ValueError("continuation did not complete its registered budget")
            write_new(
                directory / "complete.json",
                dict(complete=True, partition=2, wam_trials=1, physics_steps=600, queries=75),
            )
        except BaseException as exc:
            write_new(
                directory / "error.json",
                dict(error=repr(exc), scientific_failure_inferred=False, automatic_retry=False),
            )
            raise
        finally:
            cleanup, errors = {}, []
            for label, process, owned, cuda in (("job", job, job_identity, ""), ("model", model, model_identity, "6")):
                try:
                    _, cleanup[label] = stop_owned(process, owned, cuda)
                except BaseException as exc:
                    errors.append(dict(process=label, error=repr(exc)))
            try:
                ownership.close()
            except BaseException as exc:
                errors.append(dict(process="owned_handles", error=repr(exc)))
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
                raise RuntimeError("continuation owned cleanup failed; inspect retained release")


if __name__ == "__main__":
    main()
