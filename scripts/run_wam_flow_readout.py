"""026 owned precision/fit/sampling stages, with physicalGPU6-only workers."""

import argparse
import datetime
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from prepare_wam_flow_readout import weight_stats
from run_wam_readout import external_identity, owned_command

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once

GPU_UUID = "GPU-7f176867-170c-b508-5c89-1cd819ad3b8e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def read(path, digest=None):
    require(digest is None or sha256(path) == digest, f"026 artifact SHA differs: {path}")
    return json.loads(Path(path).read_text())


def check_hashes(mapping):
    for path, digest in mapping.items():
        require(sha256(path) == digest, f"026 source/cache SHA differs: {path}")


def read_prepared(path, digest):
    prepared = read(path, digest)
    require(prepared["schema_version"] == "s2a.flow-readout-preparation.v1", "026 preparation schema differs")
    for key in ("input_sha256", "source_sha256", "snapshot_sha256"):
        check_hashes(prepared[key])
    weight_stats(prepared["weight_files"])
    root = Path(prepared["output_root"])
    require(
        root == (REPO / prepared["config"]["output_dir"]).resolve() and root.is_relative_to(REPO / "outputs"),
        "026 output root differs",
    )
    require(
        external_identity(prepared["config"]["cosmos_repo"]) == prepared["external_cosmos"],
        "026 external source changed",
    )
    return prepared


def resource_observation():
    raw = subprocess.check_output(
        ["nvidia-smi", "--id=6", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
    physical, uuid, memory = [value.strip() for value in raw.split(",")]
    require(physical == "6" and uuid == GPU_UUID, "026 physicalGPU6 UUID differs")
    apps = subprocess.check_output(
        ["nvidia-smi", "--id=6", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    active = [
        line for line in apps.splitlines() if len(line.split(",")) >= 2 and line.split(",")[1].strip() == GPU_UUID
    ]
    return dict(
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        physical_gpu=6,
        uuid=uuid,
        memory_mib=int(memory),
        compute_apps=active,
        raw_query=raw,
        raw_compute_query=apps,
    )


def wait_release(directory, timeout=120):
    started = time.monotonic()
    with (directory / "resource_observations.jsonl").open("x") as stream:
        while True:
            observed = resource_observation()
            observed["seconds_since_actual_owned_wait"] = time.monotonic() - started
            stream.write(json.dumps(observed) + "\n")
            stream.flush()
            if observed["memory_mib"] == 0 and not observed["compute_apps"]:
                observed["released"] = True
                write_once(directory / "gpu_after.json", observed)
                return reference(directory / "gpu_after.json")
            if time.monotonic() - started >= timeout:
                raise TimeoutError("026 owned worker exited; GPU6 release was not observed within120seconds")
            time.sleep(0.1)


def environment(prepared, gpu):
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="6" if gpu else "",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=prepared["config"]["cosmos_repo"] + ":" + str(REPO / "src"),
        MASTER_PORT="31066",
        COSMOS_TRAINING="0",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        TORCHINDUCTOR_CACHE_DIR=str(Path(prepared["output_root"]) / "model_cache"),
    )


def completed(path, prepared_path):
    result = read(path)
    require(
        result.get("complete") is True and result.get("passed") is True and "error" not in result,
        "026 prior stage incomplete",
    )
    require(result["prepared_ref"] == reference(prepared_path), "026 prior stage bound to another preparation")
    return result


def verified(path, phase, prepared_path, producer_path):
    result = read(path)
    require(
        result.get("passed") is True and result.get("phase") == phase and "error" not in result,
        "026 independent audit did not pass",
    )
    require(result["prepared"] == reference(prepared_path), "026 audit bound to another preparation")
    check_hashes(result["input_sha256"])
    require(
        result["input_sha256"].get(str(Path(producer_path).resolve())) == sha256(producer_path),
        "026 audit omitted its actual producer result",
    )
    return result


def owner_completed(root, stage, prepared_path):
    result = completed(root / "orchestration" / stage / "result.json", prepared_path)
    require(
        result["stage"] == stage and len(result["owned_commands"]) == (1 if stage == "precision" else 2),
        "026 owner command budget differs",
    )
    for item in result["owned_commands"]:
        launch, exit_record, release = (
            read(item[key]["path"], item[key]["sha256"]) for key in ("launch", "exit", "release")
        )
        require(
            exit_record == dict(returncode=0, actual_wait=True, automatic_retry=False),
            "026 prior command has no successful actual wait",
        )
        require(
            all(release[key] == launch["child"][key] for key in ("pid", "start_ticks"))
            and release["reaped"] is True
            and release["returncode"] == 0
            and release["original_error"] is None
            and release["signal_sent"] is False,
            "026 prior owned command did not exit normally",
        )
    if stage != "fit":
        released = read(result["gpu_after_ref"]["path"], result["gpu_after_ref"]["sha256"])
        require(
            released["released"] is True and released["memory_mib"] == 0 and not released["compute_apps"],
            "026 GPU release proof differs",
        )
    return result


def sampling_admission(fit, audit, prepared_path, fit_path):
    require(
        fit.get("complete") is True and fit.get("passed") is True and fit["prepared_ref"] == reference(prepared_path),
        "026 fit incomplete or unrelated",
    )
    require(
        fit["producer_fit_groups"] == fit["fit_attempts"] == 2 and fit["fitted_rows"] == fit["validation_rows"] == 2880,
        "026 two-fit/whole-episode budget differs",
    )
    require(
        fit.get("eligible_for_sampling") is True and fit["teacher_gate"]["passed"] is True,
        "026 producer teacher gate rejected sampling",
    )
    require(
        audit.get("fit_verified") is True
        and audit.get("eligible_for_sampling") is True
        and audit["teacher_gate"]["passed"] is True,
        "026 independent teacher gate rejected sampling",
    )
    require(
        audit["prepared"] == reference(prepared_path)
        and audit["input_sha256"].get(str(Path(fit_path).resolve())) == sha256(fit_path),
        "026 independent gate binding differs",
    )
    for model in fit["models"].values():
        require(sha256(model["path"]) == model["sha256"], "026 fitted coefficients changed")


def planned_commands(args, prepared):
    root = Path(prepared["output_root"])
    py, model_py = str(REPO / ".venv/bin/python"), str(Path(prepared["config"]["cosmos_repo"]) / ".venv/bin/python")
    common = ["--prepared", str(args.prepared.resolve()), "--prepared-sha256", args.prepared_sha256]
    precision, fit, fit_verified = (
        root / "precision/result.json",
        root / "fit/fit_result.json",
        root / "review/fit_verified.json",
    )
    if args.stage == "precision":
        return [model_py, "scripts/probe_wam_flow_readout_quantization.py", *common, "--seed", "42"], None, precision
    if args.stage == "fit":
        worker = [
            py,
            "scripts/fit_wam_flow_readout.py",
            *common,
            "--quantization-admission",
            str(precision),
            "--quantization-admission-sha256",
            sha256(precision),
            "--seed",
            "42",
        ]
        result = fit
    else:
        worker = [
            model_py,
            "scripts/run_wam_flow_readout_sampling.py",
            *common,
            "--fit",
            str(fit),
            "--fit-sha256",
            sha256(fit),
            "--fit-verification",
            str(fit_verified),
            "--fit-verification-sha256",
            sha256(fit_verified),
            "--seed",
            "42",
        ]
        result = root / "sampling/result.json"
    audit = [
        py,
        "scripts/verify_wam_flow_readout.py",
        "--phase",
        args.stage,
        "--prepared",
        str(args.prepared.resolve()),
        "--fit",
        str(fit),
    ]
    if args.stage == "sampling":
        audit += ["--fit-verification", str(fit_verified), "--sampling", str(result)]
    audit += ["--output", str(root / "review" / f"{args.stage}_verified.json"), "--seed", "42"]
    return worker, audit, result


def execute(args):
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "026 owner requires S2A cwd and empty CUDA/LD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    peer = read(args.source_peer, args.source_peer_sha256)
    require(
        peer.get("passed") is True
        and peer.get("phase") == "prepare"
        and peer.get("preparation_verified") is True
        and peer["prepared"] == reference(args.prepared),
        "026 prepare-only source peer did not admit this preparation",
    )
    check_hashes(peer["input_sha256"])
    require(str(Path(__file__).resolve()) in prepared["source_sha256"], "026 preparation omitted its actual owner")
    root = Path(prepared["output_root"])
    if args.stage != "precision":
        owner_completed(root, "precision", args.prepared)
        completed(root / "precision/result.json", args.prepared)
    if args.stage == "sampling":
        owner_completed(root, "fit", args.prepared)
        fit_path = root / "fit/fit_result.json"
        audit = verified(root / "review/fit_verified.json", "fit", args.prepared, fit_path)
        sampling_admission(completed(fit_path, args.prepared), audit, args.prepared, fit_path)
    worker, audit, producer = planned_commands(args, prepared)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    stage=args.stage,
                    worker=worker,
                    audit=audit,
                    new_model_calls=0,
                    physical_gpu=6 if args.stage != "fit" else None,
                )
            )
        )
        return
    directory = root / "orchestration" / args.stage
    directory.parent.mkdir(exist_ok=True)
    directory.mkdir()
    require(not producer.exists() and not (root / args.stage).exists(), "026 refuses existing producer output")
    result = dict(
        schema_version="s2a.flow-readout-owner.v1",
        stage=args.stage,
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        source_peer=reference(args.source_peer),
        owned_commands=[],
        automatic_retry=False,
    )
    try:
        if args.stage == "fit":
            command_ref = owned_command(worker, environment(prepared, False), directory / "producer")
        else:
            lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks/physical6.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                before = resource_observation()
                write_once(directory / "gpu_before.json", before)
                require(
                    before["memory_mib"] == 0 and not before["compute_apps"],
                    "026 GPU6 occupied; existing processes untouched",
                )
                if args.stage == "sampling":
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 31066))
                read_prepared(args.prepared, args.prepared_sha256)
                command_ref = owned_command(
                    worker,
                    environment(prepared, True),
                    directory / "model",
                    started=root / args.stage / "worker_started.json",
                )
                result["gpu_after_ref"] = wait_release(directory)
                result["resource_observations_ref"] = reference(directory / "resource_observations.jsonl")
        result["owned_commands"].append(command_ref)
        if args.stage == "precision":
            raw = completed(root / "precision/worker_result.json", args.prepared)
            raw.update(
                exit_ref=command_ref["exit"],
                release_ref=command_ref["release"],
                gpu_after_ref=result["gpu_after_ref"],
                resource_observations_ref=result["resource_observations_ref"],
            )
            write_once(producer, raw)
        else:
            completed(producer, args.prepared)
            command_ref = owned_command(audit, environment(prepared, False), directory / "independent")
            result["owned_commands"].append(command_ref)
            checked = verified(root / "review" / f"{args.stage}_verified.json", args.stage, args.prepared, producer)
            if args.stage == "fit":
                result["eligible_for_sampling"] = bool(
                    read(producer)["eligible_for_sampling"] and checked["eligible_for_sampling"]
                )
        read_prepared(args.prepared, args.prepared_sha256)
        check_hashes(peer["input_sha256"])
        result.update(complete=True, passed=True, producer_result=reference(producer))
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_once(directory / "result.json", result)
    print(json.dumps(dict(stage=args.stage, result=reference(directory / "result.json"))))


def interrupted(signum, frame):
    raise InterruptedError(f"026 owner interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("precision", "fit", "sampling"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--source-peer", type=Path, required=True)
    parser.add_argument("--source-peer-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    execute(args)


if __name__ == "__main__":
    main()
