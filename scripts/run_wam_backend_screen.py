"""013: gate a single48-query flash3_video screen on012 release and011 SDPA35."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from run_wam_sampler_steps import check_weights, stage_admission, validate_prepared
from run_wam_short_replan import cpu_environment, process_record, stop_owned, write_new

from s2a.artifacts import REPO, sha256
from s2a.sampler_trace import check_sources

CONFIG = REPO / "configs/wam_backend_flash3_video_20261021.json"
PLAN = REPO / "plans/v0.0.1/013_attention_backend_state_20261021.md"
SDPA = REPO / "outputs/wam_stability_20261021/sampler_steps_state"
SDPA_PREPARED = SDPA / "orchestration/prepared.json"
DEST = REPO / "outputs/wam_stability_20261021/backend_state/8400_flash3_video"
PREPARED = DEST / "orchestration/prepared.json"
G3 = REPO / "outputs/wam_stability_20261021/screening/8400_g30/orchestration"


def read(path):
    return json.loads(Path(path).read_text())


def configuration():
    c = read(CONFIG)
    old = read(REPO / "configs/wam_sampler_steps_35_20261021.json")
    omit = {"attention_backend", "output_dir", "notes", "checkpoints"}
    if {k: v for k, v in c.items() if k not in omit} != {k: v for k, v in old.items() if k not in omit}:
        raise ValueError("013 differs from01135 outside backend/output/resources")
    ck, original = c["checkpoints"][0], old["checkpoints"][0]
    if len(c["checkpoints"]) != 1 or {k: v for k, v in ck.items() if k not in {"gpu", "port"}} != {
        k: v for k, v in original.items() if k not in {"gpu", "port"}
    }:
        raise ValueError("checkpoint differs from01135")
    if c["attention_backend"] != "flash3_video" or (ck["gpu"], ck["port"]) != (7, 9587):
        raise ValueError("only registered flash3_video GPU7/9587 is permitted")
    if (REPO / c["output_dir"]).resolve() != DEST:
        raise ValueError("013 output differs")
    frozen = subprocess.check_output(["git", "show", f"75c7cf1:{CONFIG.relative_to(REPO)}"])
    if sha256(CONFIG) != hashlib.sha256(frozen).hexdigest():
        raise ValueError("013 configuration changed after registration")
    return c


def prior_gates(guidance_audit):
    """No model/physical work; all prerequisites have to be complete first."""
    audit = read(guidance_audit)
    if (
        audit["schema_version"] != "s2a.guidance-state-readonly-audit.v1"
        or not audit["complete"]
        or audit["verified_queries"] != 48
        or not audit["resources"]["owned_service_released_normally"]
    ):
        raise ValueError("012 lacks complete48 independent audit and normal release")
    check_sources(audit["input_sha256"])
    released = read(G3 / "release.json")
    if (
        not read(G3 / "complete.json")["complete"]
        or released["job_returncode"] != 0
        or released["model_returncode"] not in (0, -15)
        or (G3 / "error.json").exists()
        or any(Path(f"/proc/{released[k]}").exists() for k in ("model_pid", "job_pid"))
    ):
        raise ValueError("012 resources remain present or did not release normally")
    sdpa = read(SDPA_PREPARED)
    validate_prepared(sdpa)
    gate_path = SDPA / "orchestration/run/admission35.json"
    gate = read(gate_path)
    if gate != stage_admission(sdpa, 35):
        raise ValueError("011 full48 traced35 admission differs from current evidence")
    if not gate["passed"] or gate["full_cases"] != 48:
        raise ValueError("011 full48 exact35 admission missing")
    return (
        sdpa,
        gate,
        {
            str(Path(guidance_audit).resolve()): sha256(guidance_audit),
            str(G3 / "complete.json"): sha256(G3 / "complete.json"),
            str(G3 / "release.json"): sha256(G3 / "release.json"),
            str(SDPA_PREPARED): sha256(SDPA_PREPARED),
            str(gate_path): sha256(gate_path),
        },
    )


def prepare(c, guidance_audit):
    if DEST.exists():
        raise FileExistsError("013 prepared root exists; no overwrite or automatic retry")
    sdpa, gate, inputs = prior_gates(guidance_audit)
    inputs.update(sdpa["input_sha256"])
    inputs.update({str(CONFIG): sha256(CONFIG), str(PLAN): sha256(PLAN)})
    traces = {}
    for name in sdpa["request_order"]:
        path = SDPA / "service/traces35" / f"{name}.json"
        if sha256(path) != gate["trace_sha256"][name]:
            raise ValueError("admitted SDPA35 trace changed")
        traces[name] = dict(path=str(path), sha256=sha256(path))
        inputs[str(path)] = sha256(path)
    sources = dict(sdpa["source_sha256"])
    for p in (
        CONFIG,
        REPO / "src/s2a/wam_backend_runtime.py",
        REPO / "scripts/serve_wam_backend.py",
        REPO / "scripts/run_wam_backend_screen.py",
        REPO / "scripts/run_wam_short_replan.py",
    ):
        sources[str(p)] = sha256(p)
    for relative in (
        "cosmos_framework/model/generator/mot/small_action_tower.py",
        "cosmos_framework/model/generator/mot/small_action_attention.py",
        "cosmos_framework/model/generator/mot/cosmos3_vfm_network.py",
        "cosmos_framework/model/attention/frontend.py",
        "cosmos_framework/model/attention/flash3/__init__.py",
        "cosmos_framework/model/attention/flash3/functions.py",
        "cosmos_framework/model/attention/flash3/checks.py",
    ):
        path = Path(c["cosmos_repo"]) / relative
        sources[str(path)] = sha256(path)
    check_sources(sources)
    check_sources(inputs)
    check_weights(sdpa["weight_files"])
    (DEST / "orchestration/source_snapshot").mkdir(parents=True)
    snapshots = {}
    for n, (path, digest) in enumerate(sources.items()):
        target = DEST / "orchestration/source_snapshot" / f"{n:03d}_{Path(path).name}"
        shutil.copyfile(path, target)
        if sha256(target) != digest:
            raise ValueError("013 source snapshot differs")
        snapshots[str(target)] = digest
    samples = copy.deepcopy(read(SDPA / "8400_steps35/samples/manifest.json"))
    samples["config"] = c
    sample_path = DEST / "samples/manifest.json"
    write_new(sample_path, samples)
    inputs[str(sample_path)] = sha256(sample_path)
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        prepared_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        registration_commits=["75c7cf1", "c08d433"],
        root=str(DEST),
        config_path=str(CONFIG),
        budget=dict(requests=48, physics_steps=0),
        physical_gpu=7,
        port=9587,
        checkpoint=sdpa["checkpoint"],
        weight_files=sdpa["weight_files"],
        unipc_source=sdpa["unipc_source"],
        request_order=sdpa["request_order"],
        cached_cases=sdpa["cached_cases"],
        baseline_traces=traces,
        guidance_audit_path=str(Path(guidance_audit).resolve()),
        source_sha256=sources,
        input_sha256=inputs,
        snapshot_sha256=snapshots,
        no_training_fixed_padding_restored=True,
        model_decoder_compile=False,
        internal_flex_kernel_compile_may_occur=True,
    )
    write_new(PREPARED, protocol)
    validate(protocol)
    return protocol


def validate(protocol):
    if protocol["root"] != str(DEST) or protocol["budget"] != dict(requests=48, physics_steps=0):
        raise ValueError("013 prepared root/budget differs")
    check_sources(protocol["source_sha256"])
    check_sources(protocol["input_sha256"])
    check_sources(protocol["snapshot_sha256"])
    check_weights(protocol["weight_files"])
    prior_gates(protocol["guidance_audit_path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--guidance-audit", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cpu_environment()
    c = configuration()
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    physical_gpu=7,
                    port=9587,
                    requests=48,
                    physics_steps=0,
                    attention="flash3_video",
                    prerequisites_checked=False,
                    model_started=False,
                    physics_started=False,
                )
            )
        )
        return
    if args.prepare_only:
        if args.guidance_audit is None:
            raise ValueError("preparation requires the complete012 independent audit path")
        protocol = prepare(c, args.guidance_audit)
        print(json.dumps(dict(prepared=str(PREPARED), sha256=sha256(PREPARED), budget=protocol["budget"])))
        return
    p, prepared_sha = read(PREPARED), sha256(PREPARED)
    validate(p)
    dest = DEST / "orchestration/run"
    dest.mkdir(exist_ok=False)
    lockdir = REPO / "outputs/wam_stability_20261021/gpu_locks"
    lockdir.mkdir(exist_ok=True)
    model = job = model_identity = job_identity = None

    def interrupted(signum, _frame):
        raise InterruptedError(f"owned013 coordinator received signal{signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    with (lockdir / "physical7.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            validate(p)
            memory, uuid = (
                subprocess.check_output(
                    ["nvidia-smi", "--id=7", "--query-gpu=memory.used,uuid", "--format=csv,noheader,nounits"], text=True
                )
                .strip()
                .split(",")
            )
            if int(memory) >= 500:
                raise RuntimeError("GPU7 occupied; no existing process was stopped")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 9587))
            for cache in ("torch_cache", "triton_cache", "extensions", "tmp"):
                (DEST / cache).mkdir(exist_ok=False)
            model_cmd = [
                str(Path(c["cosmos_repo"]) / ".venv/bin/python"),
                "scripts/serve_wam_backend.py",
                "--prepared",
                str(PREPARED),
                "--prepared-sha256",
                prepared_sha,
            ]
            write_new(
                dest / "launch.json",
                dict(
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    coordinator=process_record(os.getpid(), ""),
                    prepared_sha256=prepared_sha,
                    model_command=model_cmd,
                    budget=p["budget"],
                    physical_gpu=7,
                    gpu_uuid=uuid.strip(),
                    port=9587,
                    memory_mib_before=int(memory),
                ),
            )
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="7",
                LD_LIBRARY_PATH="",
                MASTER_PORT="31987",
                COSMOS_TRAINING="0",
                COSMOS_FA3_PROVIDER="nv",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=c["cosmos_repo"] + ":" + str(REPO / "src"),
                OMP_NUM_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
                TORCHINDUCTOR_CACHE_DIR=str(DEST / "torch_cache"),
                TRITON_CACHE_DIR=str(DEST / "triton_cache"),
                TORCH_EXTENSIONS_DIR=str(DEST / "extensions"),
                TMPDIR=str(DEST / "tmp"),
            )
            with (dest / "service.log").open("x") as log:
                model = subprocess.Popen(
                    model_cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            model_identity = process_record(model.pid, "7")
            if not model_identity["present"]:
                raise RuntimeError("owned model exited immediately")
            write_new(dest / "model_process.json", dict(model_identity, command=model_cmd))
            with requests.Session() as session:
                session.trust_env = False
                for _ in range(180):
                    if model.poll() is not None:
                        raise RuntimeError("013 owned model exited before readiness")
                    try:
                        response = session.get("http://127.0.0.1:9587/info", timeout=2)
                        response.raise_for_status()
                        info = response.json()
                        break
                    except requests.RequestException:
                        time.sleep(5)
                else:
                    raise TimeoutError("013 readiness timeout")
            if (
                info["checkpoint_path"] != p["checkpoint"]
                or info["ema_weights"]
                or info["attention_backend"] != "flash3_video"
                or not info["acceptance_intervention"]["active_slot_mask"]
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
            ):
                raise ValueError("013 serving identity differs")
            device_uuid = info["acceptance_intervention"]["actual_backend_probe"]["cuda_device_uuid"]
            if device_uuid.lower().removeprefix("gpu-") != uuid.strip().lower().removeprefix("gpu-"):
                raise ValueError("actual CUDA device UUID differs from the registered physical GPU7")
            write_new(dest / "server_info.json", info)
            cmd = [
                str(REPO / ".venv/bin/python"),
                "scripts/run_wam_state_matrix.py",
                "--config",
                str(CONFIG),
                "--seed",
                "42",
                "--checkpoint",
                "8400_masked",
            ]
            with (dest / "state.log").open("x") as log:
                job = subprocess.Popen(
                    cmd,
                    cwd=REPO,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            job_identity = process_record(job.pid, "")
            write_new(dest / "state_process.json", dict(job_identity, command=cmd))
            if job.wait() != 0:
                raise RuntimeError("013 state matrix failed; no automatic retry")
            validate(p)
            if sha256(PREPARED) != prepared_sha:
                raise ValueError("prepared protocol changed")
            complete_path = DEST / "state_queries/8400_masked/complete.json"
            complete = read(complete_path)
            if complete["cases"] != 48 or set(complete["result_files"]) != set(p["request_order"]):
                raise ValueError("013 full48 budget incomplete")
            trace_sha = {}
            for name in p["request_order"]:
                path = DEST / "service/traces35" / f"{name}.json"
                trace = read(path)
                if (
                    not trace["complete"]
                    or trace["velocity_evaluation_count"] != 35
                    or not trace["actual_state_and_condition_pair_with_sdpa35"]
                ):
                    raise ValueError("013 actual pair/NFE trace incomplete")
                trace_sha[name] = sha256(path)
            write_new(
                dest / "complete.json",
                dict(
                    complete=True,
                    requests=48,
                    physics_steps=0,
                    state_complete_sha256=sha256(complete_path),
                    trace_sha256=trace_sha,
                ),
            )
        except BaseException as exc:
            write_new(
                dest / "error.json", dict(error=repr(exc), scientific_failure_inferred=False, automatic_retry=False)
            )
            raise
        finally:
            cleanup_errors = []
            job_code = job_stop = model_code = model_stop = None
            try:
                job_code, job_stop = stop_owned(job, job_identity, "")
            except BaseException as exc:
                cleanup_errors.append(dict(process="owned_state_worker", error=repr(exc)))
            try:
                model_code, model_stop = stop_owned(model, model_identity, "7")
            except BaseException as exc:
                cleanup_errors.append(dict(process="owned_model", error=repr(exc)))
            write_new(
                dest / "release.json",
                dict(
                    model_pid=model.pid if model else None,
                    model_start_ticks=model_identity.get("start_ticks") if model_identity else None,
                    model_returncode=model_code,
                    job_pid=job.pid if job else None,
                    job_returncode=job_code,
                    model_stop=model_stop,
                    job_stop=job_stop,
                    cleanup_errors=cleanup_errors,
                    only_owned_process_groups_considered=True,
                ),
            )
            if cleanup_errors:
                raise RuntimeError("owned013 cleanup was incomplete; inspect release.json")


if __name__ == "__main__":
    main()
