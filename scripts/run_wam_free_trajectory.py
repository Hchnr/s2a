"""Own029's synthetic qualification and finite no-fit trajectory diagnostic."""

import argparse
import fcntl
import json
import os
import signal
import socket
from pathlib import Path

from prepare_wam_flow_readout import reference, require
from prepare_wam_free_trajectory import BACKEND_CONTRACT, read_prepared
from run_wam_flow_readout import check_hashes, completed, environment, read, resource_observation, wait_release
from run_wam_flow_readout import verified as verified_receipt
from run_wam_readout import owned_command

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once


def owner_completed(root, stage, prepared_path):
    result = completed(root / "orchestration" / stage / "result.json", prepared_path)
    require(
        result["schema_version"] == "s2a.free-trajectory-owner.v1"
        and result["stage"] == stage
        and result["automatic_retry"] is False
        and len(result["owned_commands"]) == 2,
        "029 prior owner incomplete or changed",
    )
    for command in result["owned_commands"]:
        launch, end, release = (read(command[k]["path"], command[k]["sha256"]) for k in ("launch", "exit", "release"))
        require(
            end == dict(actual_wait=True, automatic_retry=False, returncode=0)
            and release["reaped"] is True
            and release["returncode"] == 0
            and release["signal_sent"] is False
            and release["original_error"] is None
            and all(launch["child"][k] == release[k] for k in ("pid", "start_ticks")),
            "029 prior command lacks successful actual wait/reap",
        )
    released = read(result["gpu_after_ref"]["path"], result["gpu_after_ref"]["sha256"])
    require(
        released["released"] is True and released["memory_mib"] == 0 and released["compute_apps"] == [],
        "029 prior resource release missing",
    )
    return result


def precision_admission(prepared, prepared_path, precision_path, precision_sha, verification_path, verification_sha):
    precision, audit = read(precision_path, precision_sha), read(verification_path, verification_sha)
    require(
        precision["schema_version"] == "s2a.free-trajectory-synthetic-precision.v1"
        and precision["complete"] is precision["passed"] is True
        and precision["prepared_ref"] == reference(prepared_path)
        and len(precision["cases"]) == 20
        and precision["backend_arithmetic_contract"] == BACKEND_CONTRACT
        and precision["numerical_addendum_ref"] == prepared["numerical_addendum_ref"],
        "029 synthetic qualification absent or unrelated",
    )
    require(
        audit["schema_version"] == "s2a.free-trajectory-independent-verification.v1"
        and audit["passed"] is audit["precision_verified"] is True
        and audit["phase"] == "precision"
        and audit["prepared"] == reference(prepared_path)
        and audit["input_sha256"].get(str(Path(precision_path).resolve())) == precision_sha
        and audit["backend_gold_verified"] is True
        and audit["eligible_for_fit"] is audit["physics_authorized"] is False,
        "029 independent synthetic qualification absent or unrelated",
    )
    check_hashes(audit["input_sha256"])
    root = Path(prepared["output_root"])
    owner_completed(root, "precision", prepared_path)
    return precision, audit


def planned_commands(args, prepared):
    root = Path(prepared["output_root"])
    model_python = str(Path(prepared["config"]["cosmos_repo"]) / ".venv/bin/python")
    common = ["--prepared", str(args.prepared.resolve()), "--prepared-sha256", args.prepared_sha256]
    precision, audit_path = root / "precision/result.json", root / "review/precision_verified.json"
    script = "probe_wam_free_trajectory_precision.py" if args.stage == "precision" else "collect_wam_free_trajectory.py"
    worker = [model_python, f"scripts/{script}", *common]
    if args.stage == "collection":
        worker += [
            "--precision",
            str(precision),
            "--precision-sha256",
            sha256(precision),
            "--precision-verification",
            str(audit_path),
            "--precision-verification-sha256",
            sha256(audit_path),
        ]
    worker += ["--seed", "42"]
    producer = root / args.stage / "result.json"
    verifier = [
        str(REPO / ".venv/bin/python"),
        "scripts/verify_wam_free_trajectory.py",
        "--phase",
        args.stage,
        "--prepared",
        str(args.prepared.resolve()),
        "--precision",
        str(precision),
    ]
    if args.stage == "collection":
        verifier += ["--precision-verification", str(audit_path), "--collection", str(producer)]
    verifier += ["--output", str(root / "review" / f"{args.stage}_verified.json"), "--seed", "42"]
    return worker, verifier, producer


def execute(args):
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "029 owner requires repository cwd and emptyCUDA/LD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    peer = read(args.source_peer, args.source_peer_sha256)
    require(
        peer["schema_version"] == "s2a.free-trajectory-independent-verification.v1"
        and peer["phase"] == "prepare"
        and peer["passed"] is peer["preparation_verified"] is True
        and peer["prepared"] == reference(args.prepared)
        and peer["eligible_for_fit"] is peer["physics_authorized"] is False,
        "029 complete independent preparation is required",
    )
    check_hashes(peer["input_sha256"])
    root = Path(prepared["output_root"])
    if args.stage == "collection":
        p, a = root / "precision/result.json", root / "review/precision_verified.json"
        precision_admission(prepared, args.prepared, p, sha256(p), a, sha256(a))
    worker, verifier, producer = planned_commands(args, prepared)
    if args.dry_run:
        print(json.dumps(dict(stage=args.stage, worker=worker, verifier=verifier, physical_gpu=6, new_model_calls=0)))
        return
    directory = root / "orchestration" / args.stage
    require(not directory.exists() and not producer.parent.exists(), "029 refuses existing stage outputs")
    directory.mkdir(parents=True)
    result = dict(
        schema_version="s2a.free-trajectory-owner.v1",
        stage=args.stage,
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        source_peer=reference(args.source_peer),
        owned_commands=[],
        automatic_retry=False,
        eligible_for_fit=False,
        physics_authorized=False,
    )
    try:
        lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks/physical6.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = resource_observation()
            write_once(directory / "gpu_before.json", before)
            require(
                before["memory_mib"] == 0 and before["compute_apps"] == [], "029 GPU6 occupied; all processes untouched"
            )
            if args.stage == "collection":
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", prepared["config"]["master_port"]))
            read_prepared(args.prepared, args.prepared_sha256)
            env = environment(prepared, True)
            env["MASTER_PORT"] = str(prepared["config"]["master_port"])
            command = owned_command(worker, env, directory / "model", started=producer.parent / "worker_started.json")
            result["owned_commands"].append(command)
            result["gpu_after_ref"] = wait_release(directory)
            result["resource_observations_ref"] = reference(directory / "resource_observations.jsonl")
        original = completed(producer.parent / "worker_result.json", args.prepared)
        original.update(
            exit_ref=command["exit"],
            release_ref=command["release"],
            gpu_after_ref=result["gpu_after_ref"],
            resource_observations_ref=result["resource_observations_ref"],
        )
        write_once(producer, original)
        result["owned_commands"].append(
            owned_command(verifier, environment(prepared, False), directory / "independent")
        )
        audit = verified_receipt(root / "review" / f"{args.stage}_verified.json", args.stage, args.prepared, producer)
        require(
            audit["eligible_for_fit"] is audit["physics_authorized"] is False,
            "029 diagnostic must not authorize fitting or physics",
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
    raise InterruptedError(f"029 owner interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("precision", "collection"), required=True)
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
