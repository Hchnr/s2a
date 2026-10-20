"""Own one010 private model and sequential alpha1/0 frozen position matrices.

The existing model server, executor and physical controller are not modified.
Dry-run performs CPU-only prerequisite/source checks and starts no children.
"""

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

from s2a.artifacts import REPO, sha256

REGISTRATION_COMMIT = "5b13ea8"
PLAN = "plans/v0.0.1/010_short_replan_wam_20261021.md"
PRIOR_GATE = "outputs/wam_stability_20261021/short_replan_preflight/prior_gates_and_registration.json"
PRIOR_GATE_SHA = "9bc60415ee3f405f1f15087c44e21c716bcfbbc76be9fdfd72d29c6e989ab749"
CHECKPOINT = "8400_masked"
MATRIX = "scripts/run_wam_position_matrix.py"
SERVER = "scripts/serve_wam_acceptance.py"
FINGERPRINTS = "outputs/wam_independent_20261020/checkpoint_fingerprints.json"


def write_new(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(REPO / "outputs"):
        raise ValueError("artifacts must stay below repository outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cpu_environment():
    if Path.cwd().resolve() != REPO:
        raise ValueError("run from the S2A repository")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU coordinator/auditor requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")


def verify_hashes(signatures):
    for filename, expected in signatures.items():
        if sha256(filename) != expected:
            raise ValueError(f"frozen input changed: {filename}")


def config_path(interval, alpha):
    return f"configs/wam_short_replan_{interval}_alpha{alpha}_20261021.json"


def configurations(signatures):
    result = {}
    for relative in [PLAN] + [config_path(i, a) for i in (8, 16) for a in (1, 0)]:
        frozen = subprocess.check_output(["git", "show", f"{REGISTRATION_COMMIT}:{relative}"], cwd=REPO)
        expected = hashlib.sha256(frozen).hexdigest()
        if sha256(REPO / relative) != expected:
            raise ValueError(f"010 registration differs from {REGISTRATION_COMMIT}: {relative}")
        signatures[str(REPO / relative)] = expected
    for interval in (8, 16):
        for alpha in (1, 0):
            c = json.loads((REPO / config_path(interval, alpha)).read_text())
            ck = c["checkpoints"][0]
            if (
                len(c["checkpoints"]) != 1
                or (ck["name"], ck["iteration"], ck["gpu"], ck["port"])
                != (CHECKPOINT, "iter_000008400", 4 if interval == 8 else 5, 9444 if interval == 8 else 9445)
                or c["seed"] != 42
                or c["sampling_seeds"] != [42]
                or c["records"] != ["task0_init0", "task1_init0", "task2_init0"]
                or (c["online_max_steps"], c["online_chunk_size"], c["online_execution_horizon"]) != (600, 40, 40)
                or c["max_observation_interval_steps"] != interval
                or (c["control_fps"], c["stable_success_last_steps"]) != (20, 40)
                or c["variants"]
                != [
                    dict(
                        name="position_hold",
                        rotation="response",
                        gripper="position_hold",
                        governor=False,
                        model_rotation_alpha=float(alpha),
                    )
                ]
            ):
                raise ValueError("010 profile/resource/budget identity differs")
            output = (REPO / c["output_dir"]).resolve()
            if not output.is_relative_to(REPO / "outputs"):
                raise ValueError("profile output escapes repository")
            if sha256(REPO / c["fixture"]) != c["fixture_sha256"]:
                raise ValueError("fixture changed")
            result[interval, alpha] = c
        omitted = {"output_dir", "notes", "variants", "model_reference_label"}
        if {k: v for k, v in result[interval, 1].items() if k not in omitted} != {
            k: v for k, v in result[interval, 0].items() if k not in omitted
        }:
            raise ValueError("shared model profiles differ outside CPU reference intervention")
    return result


def weight_stats(gate):
    checked = []
    for item in gate["weight_stat_checks"]:
        doc = json.loads(Path(item["fingerprint_path"]).read_text())
        entry = next(x for x in doc["checkpoints"] if x["name"] == item["name"]) if "checkpoints" in doc else doc
        if len(entry["files"]) != item["files_stat_rechecked"]:
            raise ValueError("incomplete previously byte-hashed checkpoint fingerprint")
        for filename, original in entry["files"].items():
            stat = Path(filename).stat()
            if stat.st_size != original["bytes"] or stat.st_mtime_ns != original["mtime_ns"]:
                raise ValueError(f"checkpoint changed since full byte hashing: {filename}")
            if len(original["sha256"]) != 64:
                raise ValueError("missing original full-byte SHA")
        checked.append(dict(name=item["name"], stat_rechecked=len(entry["files"]), full_weight_bytes_rehashed=False))
    if {x["name"] for x in checked} != {"8400_masked", "8800_masked", "15400_masked", "15600_masked"}:
        raise ValueError("old/latest checkpoint gates incomplete")
    return checked


def process_record(pid, expected_cuda, *, start_ticks=None):
    proc = Path(f"/proc/{pid}")
    try:
        ticks = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
        env = dict(x.split(b"=", 1) for x in (proc / "environ").read_bytes().split(b"\0") if b"=" in x)
    except FileNotFoundError:
        return dict(pid=pid, present=False)
    if start_ticks is not None and ticks != start_ticks:
        return dict(pid=pid, present=False, pid_reused=True, observed_start_ticks=ticks)
    if env.get(b"CUDA_VISIBLE_DEVICES") != expected_cuda.encode() or env.get(b"LD_LIBRARY_PATH") != b"":
        raise ValueError(f"owned process environment differs: {pid}")
    return dict(pid=pid, present=True, start_ticks=ticks, cuda_visible_devices=expected_cuda, ld_library_path="")


def preflight():
    cpu_environment()
    if sha256(REPO / PRIOR_GATE) != PRIOR_GATE_SHA:
        raise ValueError("independent prerequisite gate changed")
    gate = json.loads((REPO / PRIOR_GATE).read_text())
    if (
        not gate["complete_prior_gates_passed"]
        or not gate["all_inputs_before_after_unchanged"]
        or gate["failures"]
        or (gate["registered_trials"], gate["registered_physics_steps"], gate["registered_query_upper_bound"])
        != (12, 7200, 678)
        or gate["budget_commit"] != REGISTRATION_COMMIT
    ):
        raise ValueError("incomplete010 prior/budget gates")
    signatures = dict(gate["input_sha256"])
    signatures[str(REPO / PRIOR_GATE)] = PRIOR_GATE_SHA
    configs = configurations(signatures)
    verify_hashes(signatures)
    stats = weight_stats(gate)
    for item in gate["normal_service_release"]:
        root = Path(item["root"])
        if (root / "error.json").exists():
            raise ValueError("previous owned service recorded an execution error")
        complete, released = (json.loads((root / name).read_text()) for name in ("complete.json", "release.json"))
        if complete != item["complete"] or released != item["release"] or released["model_returncode"] not in (0, -15):
            raise ValueError("previous owned service completion/release differs")
    if any(Path(f"/proc/{p['pid']}").exists() for p in gate["old_registered_processes"]):
        raise ValueError("old registered PID present; no existing process will be stopped")
    weights = next(x for x in json.loads((REPO / FINGERPRINTS).read_text())["checkpoints"] if x["name"] == CHECKPOINT)
    ck = configs[8, 1]["checkpoints"][0]
    if Path(weights["checkpoint"]).resolve() != (Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model").resolve():
        raise ValueError("8400 fingerprint identifies another model")
    signatures[str(Path(__file__).resolve())] = sha256(__file__)
    return configs, gate, signatures, stats


def commands(interval, configs):
    c = configs[interval, 1]
    service = [
        str(Path(c["cosmos_repo"]) / ".venv/bin/python"),
        SERVER,
        "--config",
        config_path(interval, 1),
        "--seed",
        "42",
        "--checkpoint",
        CHECKPOINT,
        "--port",
        str(c["checkpoints"][0]["port"]),
        "--active-slot-mask",
    ]
    jobs = [
        dict(
            alpha=alpha,
            profile=f"interval{interval}_alpha{alpha}",
            config=config_path(interval, alpha),
            command=[
                str(REPO / ".venv/bin/python"),
                MATRIX,
                "--config",
                config_path(interval, alpha),
                "--seed",
                "42",
                "--source",
                "wam",
                "--variant",
                "position_hold",
                "--checkpoint",
                CHECKPOINT,
            ],
        )
        for alpha in (1, 0)
    ]
    return service, jobs


def stop_owned(process, identity, expected_cuda):
    if process is None:
        return None, None
    if identity is None or "start_ticks" not in identity:
        return process.poll(), dict(identity_unavailable=True, signal_sent=False)
    observed = process_record(process.pid, expected_cuda, start_ticks=identity["start_ticks"])
    if observed.get("pid_reused"):
        return process.poll(), dict(observed, signal_sent=False)
    if process.poll() is None:
        if not observed["present"]:
            return process.wait(timeout=10), dict(observed, signal_sent=False)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=10)
            return process.returncode, dict(observed, signal_sent=False)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    return process.returncode, dict(observed, signal_sent=observed["present"])


def interrupted(signum, _frame):
    raise InterruptedError(f"owned coordinator received signal{signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, choices=(8, 16), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-output", type=Path, help="new optional CPU-only preflight artifact")
    args = parser.parse_args()
    configs, gate, signatures, stats = preflight()
    c = configs[args.interval, 1]
    ck = c["checkpoints"][0]
    gpu, port = ck["gpu"], ck["port"]
    model_command, jobs = commands(args.interval, configs)
    service_root = str((REPO / c["output_dir"] / "intervention_services" / str(port)).resolve())
    summary = dict(
        registration_commit=REGISTRATION_COMMIT,
        global_budget=dict(wam_trials=12, physics_steps=7200, model_queries_upper_bound=678, added_oracle_trials=0),
        interval=args.interval,
        physical_gpu=gpu,
        port=port,
        service_command=model_command,
        jobs=jobs,
        service_artifact_root_for_both_profiles=service_root,
        prerequisite_gate_sha256=PRIOR_GATE_SHA,
        rechecked_input_count=len(signatures),
        weight_stat_checks=stats,
        model_started=False,
        physics_started=False,
    )
    if args.preflight_output:
        write_new(args.preflight_output, dict(summary, source_sha256=signatures))
    if args.dry_run:
        print(json.dumps(summary))
        return
    root = REPO / c["output_dir"]
    directory = root / "orchestration" / f"short_replan_interval{args.interval}"
    directory.mkdir(parents=True, exist_ok=False)
    for job in jobs:
        output = REPO / configs[args.interval, job["alpha"]]["output_dir"]
        if (output / "simulation").exists():
            raise FileExistsError(f"refusing to repeat existing physics matrix: {output}")
    if Path(service_root).exists():
        raise FileExistsError("refusing to overwrite an earlier service")
    locks = REPO / "outputs/wam_stability_20261021/gpu_locks"
    locks.mkdir(parents=True, exist_ok=True)
    model = job_process = None
    model_identity = worker = None
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    with (locks / f"physical{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify_hashes(signatures)
            weight_stats(gate)
            memory = subprocess.check_output(
                ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
            ).strip()
            if int(memory) >= 500:
                raise RuntimeError("allowed GPU occupied; no existing process was stopped")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
            identity = process_record(os.getpid(), "")
            write_new(
                directory / "manifest.json",
                dict(
                    schema_version="s2a.short-replan-coordinator.v1",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    **summary,
                    coordinator=identity,
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    configurations={str(alpha): configs[args.interval, alpha] for alpha in (1, 0)},
                    frozen_input_sha256=signatures,
                    gpu_memory_mib_before=int(memory),
                    profile_service_binding={
                        j["profile"]: dict(
                            service_config=config_path(args.interval, 1),
                            service_config_sha256=sha256(REPO / config_path(args.interval, 1)),
                            service_artifact_root=service_root,
                            cpu_config=j["config"],
                            cpu_config_sha256=sha256(REPO / j["config"]),
                        )
                        for j in jobs
                    },
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
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=REPO,
                )
                model_identity = process_record(model.pid, str(gpu))
                if not model_identity["present"]:
                    raise RuntimeError("owned model exited immediately")
                write_new(
                    directory / "model_process.json", dict(model_identity, gpu=gpu, port=port, command=model_command)
                )
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
                        raise TimeoutError("owned model readiness timeout; no automatic retry")
                expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
                if (
                    Path(info["checkpoint_path"]).resolve() != expected.resolve()
                    or Path(info["config_file"]).resolve() != (Path(ck["run"]) / "config.yaml").resolve()
                    or info["ema_weights"]
                    or info["port"] != port
                    or not info["acceptance_intervention"]["active_slot_mask"]
                    or info["acceptance_intervention"]["exact_published_caption"]
                    or info["acceptance_intervention"]["warm_start_contract_run"] is not None
                ):
                    raise ValueError("owned model identity/mask/intervention mismatch")
                for relative in (SERVER, "src/s2a/wam_runtime.py"):
                    if (
                        info["acceptance_intervention"]["code"]["files_sha256"][relative]
                        != signatures[str(REPO / relative)]
                    ):
                        raise ValueError("serving source differs from frozen gate")
                write_new(directory / "server_info.json", info)
                for job in jobs:
                    verify_hashes(signatures)
                    weight_stats(gate)
                    if model.poll() is not None:
                        raise RuntimeError("owned model exited before next matrix")
                    if not process_record(model.pid, str(gpu), start_ticks=model_identity["start_ticks"])["present"]:
                        raise RuntimeError("owned model identity disappeared before next matrix")
                    with (directory / f"{job['profile']}.log").open("x") as log:
                        job_process = subprocess.Popen(
                            job["command"],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            cwd=REPO,
                        )
                        worker = process_record(job_process.pid, "")
                        if not worker["present"]:
                            raise RuntimeError("owned CPU matrix exited before identity was recorded")
                        write_new(directory / f"{job['profile']}_process.json", dict(worker, command=job["command"]))
                        code = job_process.wait()
                        write_new(directory / f"{job['profile']}_exit.json", dict(pid=job_process.pid, returncode=code))
                        if code:
                            raise RuntimeError(
                                f"physics matrix failed:{job['profile']}; no scientific conclusion inferred"
                            )
                    profile_root = REPO / configs[args.interval, job["alpha"]]["output_dir"]
                    queries = (600 + args.interval - 1) // args.interval
                    for record in c["records"]:
                        case = profile_root / "simulation" / f"{CHECKPOINT}_position_hold" / f"{record}_s42"
                        result = json.loads((case / "result.json").read_text())
                        if (
                            not result["complete"]
                            or result["error"] is not None
                            or result["steps"] != 600
                            or result["chunks"] != queries
                            or len(list(case.glob("request_*.json"))) != queries
                        ):
                            raise ValueError("finished matrix differs from fixed010 case/query budget")
                    verify_hashes(signatures)
                    weight_stats(gate)
                write_new(
                    directory / "complete.json",
                    dict(
                        complete=True,
                        profiles=[j["profile"] for j in jobs],
                        wam_trials=6,
                        physics_steps=3600,
                        queries_expected=6 * ((600 + args.interval - 1) // args.interval),
                    ),
                )
        except BaseException as exc:
            write_new(
                directory / "error.json",
                dict(error=repr(exc), scientific_failure_inferred=False, automatic_retry=False),
            )
            raise
        finally:
            job_code, job_stop = stop_owned(job_process, worker, "")
            model_code, model_stop = stop_owned(model, model_identity, str(gpu))
            write_new(
                directory / "release.json",
                dict(
                    model_pid=model.pid if model else None,
                    model_start_ticks=model_identity["start_ticks"] if model_identity else None,
                    model_returncode=model_code,
                    last_job_pid=job_process.pid if job_process else None,
                    last_job_returncode=job_code,
                    job_stop_identity=job_stop,
                    model_stop_identity=model_stop,
                    model_absent=(
                        not process_record(model.pid, str(gpu), start_ticks=model_identity["start_ticks"])["present"]
                    )
                    if model is not None and model_identity is not None
                    else model is None,
                    only_owned_process_groups_stopped=True,
                    automatic_retry=False,
                ),
            )


if __name__ == "__main__":
    main()
