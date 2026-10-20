"""Own one registered017 raw8 long-duration development partition."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from run_wam_long_matrix import protocol
from run_wam_short_replan import cpu_environment, interrupted, process_record, stop_owned, verify_hashes, write_new

from s2a.artifacts import REPO, sha256


def config_path(partition):
    return f"configs/wam_long_raw_task{partition}_20261021.json"


MATRIX = "scripts/run_wam_long_matrix.py"
SERVER = "scripts/serve_wam_acceptance.py"
FINGERPRINT = "outputs/wam_independent_20261020/checkpoint_fingerprints.json"
FINGERPRINT_SHA = "624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8"
PRIOR = REPO / "outputs/wam_stability_20261021/long_raw_preflight/prior_services_verified.json"
PRIOR_SHA = "698d0171234814b787148ddde8b1abaad574de3caacb9f5de9109362d2c54ef7"


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


def preflight(path, digest, partition):
    actual_config = config_path(partition)
    registered_config = subprocess.check_output(["git", "show", f"9fe490a:{actual_config}"], cwd=REPO)
    if sha256(REPO / actual_config) != hashlib.sha256(registered_config).hexdigest():
        raise ValueError("017 actual configuration differs from its registered bytes")
    path = Path(path).resolve()
    if not path.is_relative_to(REPO / "outputs") or sha256(path) != digest:
        raise ValueError("launch gate must be the explicit unchanged repository artifact")
    gate = json.loads(path.read_text())
    if gate["schema_version"] != "s2a.long-raw-launch-gate.v1" or not gate["all_prerequisites_passed"]:
        raise ValueError("017 launch prerequisites are incomplete")
    c, root, records, ck = protocol(actual_config, source="wam", variant_name="position_hold", checkpoint="8400_masked")
    if (
        ck["gpu"] != 5 + partition
        or ck["port"] != 9655 + partition
        or c["records"] != [f"task{partition}_init0"]
        or c["sampling_seeds"] != [42]
        or (c["online_max_steps"], c["sampling_steps"], c["guidance"], c["shift"]) != (1200, 35, 1, 5)
        or c["control_fps"] != 20
        or c["stable_success_last_steps"] != 40
        or c["max_observation_interval_steps"] != 8
        or c["episode_horizon"] != 1300
        or c["ignore_done"] is not False
    ):
        raise ValueError("017 fixed partition/resource/old-scene identity differs")
    signatures = dict(gate["input_sha256"])
    if signatures.get(str(REPO / actual_config)) != sha256(REPO / actual_config):
        raise ValueError("launch gate omitted actual configuration")
    for relative in (
        "plans/v0.0.1/017_long_raw_development_20261021.md",
        "src/s2a/long_environment.py",
        "src/s2a/long_simulation.py",
        MATRIX,
        SERVER,
        "src/s2a/wam_runtime.py",
        "scripts/run_wam_short_replan.py",
        "scripts/run_wam_long_service.py",
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
        or not qualification["normal_process_release"]
        or qualification["native_trials"] != 3
        or qualification["oracle_trials"] != 3
        or qualification["physical_steps"] != 7200
    ):
        raise ValueError("017 six-case independent execution qualification is incomplete")
    for name, digest_value in qualification["input_sha256"].items():
        if signatures.get(name) != digest_value:
            raise ValueError("launch gate omitted a qualification input")
    prior = gate["prior_release"]
    if signatures.get(str(PRIOR)) != PRIOR_SHA or sha256(PRIOR) != PRIOR_SHA:
        raise ValueError("launch gate omitted the independently checked prior-service roster")
    registered_prior = json.loads(PRIOR.read_text())
    if not registered_prior["prior_scopes_passed"] or prior != registered_prior["partitions"][str(partition)]:
        raise ValueError("prior release differs from its independently checked complete process roster")
    for filename, expected in registered_prior["input_sha256"].items():
        if signatures.get(filename) != expected:
            raise ValueError("launch gate omitted an independently checked prior-service input")
    if prior["physical_gpu"] != ck["gpu"] or not prior["whole_same_gpu_budget_complete"]:
        raise ValueError("prior same-card registered budget is incomplete")
    for kind in ("complete", "release", "audit"):
        name, expected = prior[f"{kind}_path"], prior[f"{kind}_sha256"]
        if signatures.get(name) != expected or sha256(name) != expected:
            raise ValueError(f"prior {kind} evidence is not bound to the launch gate")
    complete = json.loads(Path(prior["complete_path"]).read_text())
    release = json.loads(Path(prior["release_path"]).read_text())
    audit = json.loads(Path(prior["audit_path"]).read_text())
    prior_inputs = audit["source_before_after_hashes" if partition == 0 else "input_sha256"]
    for filename, expected in prior_inputs.items():
        if signatures.get(filename) != expected:
            raise ValueError("launch gate omitted an independent prior-audit input")
    if (
        not complete["complete"]
        or release["model_returncode"] not in (0, -15)
        or release["last_job_returncode"] != 0
        or release.get("cleanup_errors", [])
        or (partition != 1 and not release["only_owned_process_groups_stopped"])
    ):
        raise ValueError("prior owned service did not complete and release normally")
    if partition == 0 and (complete["wam_trials"], complete["physics_steps"], complete["queries_expected"]) != (
        6,
        3600,
        228,
    ):
        raise ValueError("010 interval16 complete budget differs")
    if partition == 0:
        selected = [case for case in audit["cases"] if case["interval"] == 16]
        expected_cases = {(f"task{task}_init0", alpha) for task in (0, 1, 2) for alpha in (0, 1)}
        if (
            len(selected) != 6
            or {(case["record"], case["model_rotation_alpha"]) for case in selected} != expected_cases
            or any(case["status"] != "verified" or (case["steps"], case["chunks"]) != (600, 38) for case in selected)
        ):
            raise ValueError("010 interval16 independent six-case audit is incomplete")
    if partition == 1 and (complete["requests"], complete["physics_steps"]) != (144, 0):
        raise ValueError("011 complete sampler budget differs")
    if partition == 1 and (
        not audit["complete"]
        or not audit["quality_evidence_complete"]
        or audit["errors"]
        or audit["scope"]["requests"] != 144
        or any(
            audit["groups"][str(stage)]["verified_requests"] != 48
            or not audit["groups"][str(stage)]["data_complete"]
            or not audit["groups"][str(stage)]["stage_admission_present"]
            for stage in (35, 70, 140)
        )
    ):
        raise ValueError("011 independent144-request quality audit is incomplete")
    if partition == 2 and (complete["wam_trials"], complete["physics_steps"]) != (3, 1800):
        raise ValueError("016 complete raw-reference budget differs")
    if partition == 2 and (
        not audit["complete"]
        or not audit["all_cases_independently_verified"]
        or audit["verified_cases"] != 3
        or len(audit["cases"]) != 3
        or any(case["status"] != "verified" or case["steps"] != 600 for case in audit["cases"])
    ):
        raise ValueError("016 independent three-case audit is incomplete")
    owned = prior["owned_processes"]
    models = [p for p in owned if p["cuda_visible_devices"] == str(ck["gpu"])]
    if (
        len(owned) < 2
        or len({p["pid"] for p in owned}) != len(owned)
        or len(models) != 1
        or models[0]["pid"] != release["model_pid"]
        or models[0]["start_ticks"] != int(release["model_startticks" if partition == 1 else "model_start_ticks"])
        or any(p["cuda_visible_devices"] not in ("", str(ck["gpu"])) for p in owned)
    ):
        raise ValueError("prior owned model PID/ticks/CUDA identity differs")
    for process in owned:
        if process_record(process["pid"], process["cuda_visible_devices"], start_ticks=process["start_ticks"])[
            "present"
        ]:
            raise RuntimeError("prior owned process is still live")
    return c, root, records, ck, signatures, weight_stats(ck)


def commands(c, ck, partition):
    actual_config = config_path(partition)
    model = [
        str(Path(c["cosmos_repo"]) / ".venv/bin/python"),
        SERVER,
        "--config",
        actual_config,
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
        actual_config,
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
    parser.add_argument("--partition", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cpu_environment()
    c, root, records, ck, signatures, stats = preflight(args.gate, args.gate_sha256, args.partition)
    model_command, job_command = commands(c, ck, args.partition)
    gpu, port = ck["gpu"], ck["port"]
    summary = dict(
        schema_version="s2a.long-raw-coordinator.v1",
        partition=args.partition,
        physical_gpu=gpu,
        port=port,
        wam_trials=1,
        physics_steps=1200,
        maximum_h40_queries=150,
        model_command=model_command,
        job_command=job_command,
        weight_stats=stats,
        source_sha256=signatures,
    )
    if args.dry_run:
        print(json.dumps(dict(summary, model_started=False, physics_started=False)))
        return
    if (root / "simulation").exists() or (root / "intervention_services").exists():
        raise FileExistsError("refusing to repeat/overwrite017 model or physics evidence")
    directory = root / "orchestration"
    directory.mkdir(parents=True, exist_ok=False)
    locks = REPO / "outputs/wam_stability_20261021/gpu_locks"
    locks.mkdir(parents=True, exist_ok=True)
    model = job = None
    model_identity = job_identity = None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    with (locks / f"physical{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            preflight(args.gate, args.gate_sha256, args.partition)
            memory = int(
                subprocess.check_output(
                    ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
                ).strip()
            )
            if memory != 0:
                raise RuntimeError(f"allowed GPU{gpu} occupied; no other process stopped")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
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
                CUDA_VISIBLE_DEVICES=str(gpu),
                LD_LIBRARY_PATH="",
                MASTER_PORT=str(30900 + gpu),
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
                model_identity = process_record(model.pid, str(gpu))
                if not model_identity["present"]:
                    raise RuntimeError("owned model exited before registration")
                write_new(directory / "model_process.json", dict(model_identity, command=model_command))
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
                        raise TimeoutError("model readiness timed out; no automatic retry")
                expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
                intervention = info["acceptance_intervention"]
                if (
                    Path(info["checkpoint_path"]).resolve() != expected.resolve()
                    or Path(info["config_file"]).resolve() != (Path(ck["run"]) / "config.yaml").resolve()
                    or info["port"] != port
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
                        or result["steps"] != 1200
                        or count != 150
                        or result["chunks"] != count
                    ):
                        raise ValueError("017 finished partition differs from one1200-step/150-query case")
                    query_count += count
                verify_hashes(signatures)
                weight_stats(ck)
                write_new(
                    directory / "complete.json",
                    dict(
                        complete=True, wam_trials=1, physics_steps=1200, queries=query_count, partition=args.partition
                    ),
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
                ("model", model, model_identity, str(gpu)),
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
