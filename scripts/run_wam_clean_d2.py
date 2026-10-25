"""056 detached D2 owner: one model child, real wait/reap and independent audit."""

import argparse
import fcntl
import json
import os
import socket
from pathlib import Path

from run_wam_clean_bridge import check_owned_command, resource_observation
from run_wam_clean_training import environment, observe_release
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_d2_execution import admit
from s2a.clean_training_runtime import GPU, publish, read, reference, require
from s2a.durable_stage_supervisor import accept_detached_launch, cpu_environment, launch_detached

PORT = 31356


def flags(name, ref):
    return ["--" + name, ref["path"], "--" + name + "-sha256", ref["sha256"]]


def worker_command(prepared, pref, regref):
    return [
        prepared["model_config"]["model_python"],
        "scripts/collect_wam_clean_d2.py",
        *flags("prepared", pref),
        *flags("execution-registration", regref),
        "--gpu-uuid",
        GPU,
        "--seed",
        "42",
    ]


def execute(pref, regref):
    prepared, registered, sources, *_ = admit(pref, regref, required_sources=[Path(__file__)])
    require(
        registered["numerical_verifier_ref"] == reference(REPO / "scripts/verify_wam_clean_d2.py"),
        "056 numerical verifier changed",
    )
    root = Path(prepared["root"])
    dry = read(reference(root / "successful_cli_dry.json"))
    require(
        dry["passed"] is True
        and dry["prepared_ref"] == pref
        and dry["command"] == worker_command(prepared, pref, regref) + ["--dry"],
        "Exact successful054 CLI dry required",
    )
    folder = root / "orchestration"
    folder.mkdir(exist_ok=False)
    result = {
        "schema_version": "s2a.clean-d2-owner.v1",
        "complete": False,
        "passed": False,
        "prepared_ref": pref,
        "registration_ref": regref,
        "owned_commands": [],
        "automatic_retry": False,
        "fresh_authorized": False,
    }
    env = environment(prepared, gpu=True)
    env["MASTER_PORT"] = str(PORT)
    try:
        probe = root / "d2_write_probe"
        with probe.open("xb") as stream:
            stream.write(b"056 D2 prelaunch write probe\n")
            stream.flush()
            os.fsync(stream.fileno())
        probe.unlink()
        space = os.statvfs(root)
        result["prelaunch_write_probe"] = dict(passed=True, available_bytes=space.f_bavail * space.f_frsize)
        require(space.f_bavail * space.f_frsize >= 20 * 1024**3, "056 insufficient available filesystem space")
        lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks" / (GPU + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = resource_observation(GPU)
            result["gpu_before_ref"] = publish(folder / "gpu_before.json", before)
            require(
                before["device_minor"] == 6
                and before["visible_gpu"] == 2
                and before["memory_mib"] == 0
                and before["compute_apps"] == []
                and before["utilization_gpu"] == 0,
                "056 allowed GPU is occupied or identity changed",
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", PORT))
            try:
                model = owned_command(worker_command(prepared, pref, regref), env, folder / "model", timeout=43200)
                result["owned_commands"].append(model)
                check_owned_command(model)
            finally:
                if (folder / "model/ownership/release.json").is_file():
                    result["gpu_after_ref"] = observe_release(folder)
        sampling = reference(root / "capture/D2/result.json")
        result["collection_ref"] = sampling
        command = [
            str(REPO / ".venv/bin/python"),
            "scripts/verify_wam_clean_d2.py",
            *flags("prepared", pref),
            *flags("execution-registration", regref),
            *flags("collection", sampling),
            *flags("worker-launch", model["launch"]),
            *flags("worker-exit", model["exit"]),
            *flags("worker-release", model["release"]),
            *flags("gpu-after", result["gpu_after_ref"]),
            "--output",
            str(root / "d2_verified.json"),
            "--seed",
            "42",
        ]
        checked = owned_command(command, cpu_environment(), folder / "independent", timeout=14400)
        result["owned_commands"].append(checked)
        check_owned_command(checked)
        numerical = reference(root / "d2_verified.json")
        audit = read(numerical)
        require(
            audit["passed"] is audit["complete"] is True
            and audit["collection_ref"] == sampling
            and audit["actual_requests"] == 300
            and audit["actual_denoise_calls"] == 10500,
            "056 independent full capture qualification required",
        )
        result.update(complete=True, passed=True, numerical_ref=numerical, fit_authorized=False)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["available_failure_receipts"] = {
            str(p.relative_to(folder)): reference(p)
            for p in folder.rglob("*.json")
            if p.name in ["launch.json", "exit.json", "release.json"]
        }
        publish(folder / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    pref, regref = reference(args.prepared), reference(args.registration)
    prepared, _, sources, *_ = admit(pref, regref, required_sources=[Path(__file__)])
    folder = Path(prepared["root"]) / "supervisor"
    command = [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_clean_d2.py",
        "--prepared",
        pref["path"],
        "--registration",
        regref["path"],
    ]
    if args.launch:
        resource = resource_observation(GPU)
        require(
            resource["memory_mib"] == 0 and resource["compute_apps"] == [] and resource["utilization_gpu"] == 0,
            "056 GPU unavailable; no owner/model launched",
        )
        print(json.dumps(launch_detached(command, directory=folder, source_sha256=sources)), flush=True)
        return
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "", "056 owner must be CPU")
    accept_detached_launch(folder, command=command, source_sha256=sources)
    execute(pref, regref)


if __name__ == "__main__":
    main()
