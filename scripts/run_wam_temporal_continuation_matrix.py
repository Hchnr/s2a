"""021 single-task continuation with a child-ready / parent-go CPU barrier.

The physical callable is imported unchanged from the frozen020 executor. This
entry replaces process admission only and never launches a model or retries.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import uuid
from pathlib import Path

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import (
    capture_cpu_identity,
    contract_identity,
    digest,
    go_record,
    identity,
    owned_child,
    self_ready,
    validate_ready,
    wait_for_go,
    wait_ready,
    write_once,
)

REGISTRATION = "ea3f26d"
CONFIG = "configs/wam_temporal_equal_task2_continuation_20261021.json"
PLAN = "plans/v0.0.1/021_temporal_zero_step_continuation_20261021.md"
EXTRA_SOURCES = ("src/s2a/worker_handshake.py", "scripts/run_wam_temporal_continuation_matrix.py")


def registered(relative):
    expected = hashlib.sha256(
        subprocess.check_output(["git", "show", f"{REGISTRATION}:{relative}"], cwd=REPO)
    ).hexdigest()
    if sha256(REPO / relative) != expected:
        raise ValueError("021 source differs from its preregistered bytes")
    return expected


def protocol(config_path):
    from s2a.temporal_ensemble_simulation import temporal_variant

    if Path(config_path).resolve() != REPO / CONFIG:
        raise ValueError("021 allows only its explicit registered continuation configuration")
    registered(CONFIG)
    registered(PLAN)
    config = json.loads((REPO / CONFIG).read_text())
    original = json.loads(
        subprocess.check_output(["git", "show", "7025c45:configs/wam_temporal_equal_task2_20261021.json"], cwd=REPO)
    )
    expected = json.loads(json.dumps(original))
    expected["output_dir"] = "outputs/wam_stability_20261021/temporal_equal_development/task2_continuation"
    expected["checkpoints"][0]["port"] = 9946
    expected["notes"] = config["notes"]
    if config != expected or config["notes"][:-1] != original["notes"]:
        raise ValueError("continuation changes more than its registered output, port and appended note")
    temporal_variant(config, "temporal_equal", source="wam")
    fixture_path = REPO / config["fixture"]
    if sha256(fixture_path) != config["fixture_sha256"]:
        raise ValueError("fixture changed")
    record = next(r for r in json.loads(fixture_path.read_text())["records"] if r["name"] == "task2_init0")
    checkpoint = config["checkpoints"][0]
    if config["records"] != [record["name"]] or checkpoint["gpu"] != 6 or checkpoint["port"] != 9946:
        raise ValueError("continuation task/GPU/port differs")
    return config, REPO / config["output_dir"], record, checkpoint


def execution_inputs(config, config_path):
    from s2a.temporal_ensemble_simulation import execution_inputs as frozen_inputs

    signatures = frozen_inputs(config, config_path)
    for relative in (*EXTRA_SOURCES, PLAN):
        path = REPO / relative
        signatures[str(path)] = sha256(path)
    return signatures


def interrupted(signum, frame):
    raise InterruptedError(f"021 matrix received signal {signum}")


def process_registry(observed, ready_path, snapshot_path, observations):
    return dict(
        observed,
        identity=contract_identity(observed),
        ready_sha256=digest(ready_path),
        ready_path=str(ready_path),
        parent_observations=observations,
        snapshot_sha256=digest(snapshot_path),
        automatic_retry=False,
    )


def matrix_trial(observed, record, code):
    return dict(observed, record=record, sampling_seed=42, returncode=code)


def worker(args, config, root, record):
    from s2a.temporal_ensemble_simulation import run_temporal_case, verify_inputs

    if not all((args.worker_snapshot, args.handshake_dir, args.handshake_token)) or args.sampling_seed != 42:
        raise ValueError("worker cannot bypass its registered parent handshake")
    snapshot_path = Path(args.worker_snapshot).resolve()
    folder = Path(args.handshake_dir).resolve()
    if not snapshot_path.is_relative_to(root) or folder != snapshot_path.parent / "task2_init0_s42.handshake":
        raise ValueError("worker handshake/snapshot must belong to this exact continuation")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot["identity"] != ["wam", "temporal_equal", "8400_masked"] or snapshot["jobs"] != [["task2_init0", 42]]:
        raise ValueError("snapshot identifies another job")
    if snapshot["config_sha256"] != sha256(args.config) or args.record != record["name"]:
        raise ValueError("worker config/record changed")
    if snapshot["coordinator_identity"]["pid"] != os.getppid():
        raise ValueError("worker was not spawned by its recorded matrix")
    ready = self_ready(token=args.handshake_token, config_path=args.config, snapshot_path=snapshot_path)
    expected_parent = {key: snapshot["coordinator_identity"][key] for key in ("pid", "start_ticks")}
    if ready["parent_identity"] != expected_parent:
        raise ValueError("worker parent birth differs from snapshot")
    write_once(folder / "ready.json", ready)
    wait_for_go(folder / "ready.json", folder / "go.json")
    verify_inputs(snapshot["execution_sha256"])
    result = run_temporal_case(
        config,
        root,
        record,
        source="wam",
        variant_name="temporal_equal",
        sampling_seed=42,
        checkpoint="8400_masked",
        config_path=args.config,
        frozen_inputs=snapshot["execution_sha256"],
    )
    print(json.dumps(result), flush=True)


def execute(config, root, record, config_path):
    """One Popen, registered before go; every exit path reaps that owned child."""
    from s2a.temporal_ensemble_simulation import verify_inputs

    logs = root / "simulation/logs/8400_masked_temporal_equal"
    logs.mkdir(parents=True, exist_ok=False)
    snapshot_path = logs / "execution_snapshot.json"
    coordinator = identity(os.getpid())
    snapshot = dict(
        identity=["wam", "temporal_equal", "8400_masked"],
        config_sha256=sha256(config_path),
        execution_sha256=execution_inputs(config, config_path),
        jobs=[[record["name"], 42]],
        coordinator_pid=os.getpid(),
        coordinator_identity=coordinator,
        model_started=False,
        cpu_workers=1,
        actual_module_scope="frozen020 physical callable plus explicit021 admission sources",
    )
    write_once(snapshot_path, snapshot)
    folder = logs / "task2_init0_s42.handshake"
    folder.mkdir(exist_ok=False)
    token = uuid.uuid4().hex
    command = [
        str((REPO / config["simulation_python"]).absolute()),
        str(Path(__file__).resolve()),
        "--config",
        str(Path(config_path).resolve()),
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
        "--record",
        record["name"],
        "--sampling-seed",
        "42",
        "--worker-snapshot",
        str(snapshot_path),
        "--handshake-dir",
        str(folder),
        "--handshake-token",
        token,
    ]
    environment = dict(
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
    observations = []
    registry_path = logs / "task2_init0_s42.process.json"
    try:
        verify_inputs(snapshot["execution_sha256"])
        with (
            (logs / "task2_init0_s42.log").open("x") as stream,
            owned_child(command, environment=environment, cwd=REPO, folder=folder, stream=stream) as (
                process,
                owned_birth,
            ),
        ):
            ready = wait_ready(process, folder / "ready.json", timeout_s=30)
            observed = capture_cpu_identity(
                process.pid,
                expected_ticks=owned_birth["start_ticks"],
                expected_ppid=os.getpid(),
                expected_command=command,
                expected_cwd=REPO,
                observations=observations,
            )
            validate_ready(
                ready,
                observed,
                token=token,
                config_path=config_path,
                snapshot_path=snapshot_path,
                parent_identity={key: coordinator[key] for key in ("pid", "start_ticks")},
            )
            registry = process_registry(observed, folder / "ready.json", snapshot_path, observations)
            write_once(registry_path, registry)
            write_once(folder / "go.json", go_record(folder / "ready.json", registry_path))
            code = process.wait()
        verify_inputs(snapshot["execution_sha256"])
        trial = matrix_trial(observed, record["name"], code)
        write_once(
            root / "simulation/temporal_matrix_8400_masked_temporal_equal.json",
            dict(
                complete=code == 0,
                cases=[trial],
                execution_snapshot_sha256=digest(snapshot_path),
                model_started=False,
            ),
        )
        if code:
            raise ChildProcessError("continuation worker failed; no retry")
    except BaseException as exc:
        write_once(
            folder / "error.json", dict(error=repr(exc), parent_observations=observations, automatic_retry=False)
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--source", choices=("wam",), required=True)
    parser.add_argument("--variant", choices=("temporal_equal",), required=True)
    parser.add_argument("--checkpoint", choices=("8400_masked",), required=True)
    parser.add_argument("--workers", type=int, choices=(1,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", help=argparse.SUPPRESS)
    parser.add_argument("--sampling-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-snapshot", help=argparse.SUPPRESS)
    parser.add_argument("--handshake-dir", help=argparse.SUPPRESS)
    parser.add_argument("--handshake-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        Path.cwd().resolve() != REPO
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("021 requires repository cwd and explicitly empty CUDA/LD")
    config, root, record, _ = protocol(args.config)
    if args.record is not None:
        if args.dry_run:
            raise ValueError("worker dry-run is forbidden")
        worker(args, config, root, record)
        return
    if any((args.sampling_seed is not None, args.worker_snapshot, args.handshake_dir, args.handshake_token)):
        raise ValueError("worker-only arguments require the registered record")
    if args.dry_run:
        print(
            json.dumps(
                dict(jobs=[[record["name"], 42]], physical_gpu=6, port=9946, model_started=False, physics_started=False)
            )
        )
        return
    if (root / "simulation/8400_masked_temporal_equal/task2_init0_s42").exists():
        raise FileExistsError("refusing to overwrite/retry an existing continuation case")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    lock_path = REPO / "outputs/wam_stability_20261021/service_physics_locks/port9946.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(config, root, record, args.config)


if __name__ == "__main__":
    main()
