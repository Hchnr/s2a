"""028 owns GPU6 precision and paired state generation, without fitting."""

import argparse
import fcntl
import json
import os
import signal
import socket
from pathlib import Path

from prepare_wam_flow_readout import weight_stats
from run_wam_flow_readout import (
    check_hashes,
    completed,
    environment,
    read,
    reference,
    require,
    resource_observation,
    wait_release,
)
from run_wam_flow_readout import verified as verified_receipt
from run_wam_readout import external_identity, owned_command

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/wam_scaled_readout_state_20261021.json"
CONFIG_SHA = "183b4a9407bbfaea8e82c1d7c052f5417e32def31805528102e3bd293a6cea26"
PLAN = REPO / "plans/v0.0.1/028_scaled_readout_state_20261021.md"
PLAN_SHA = "cfe26406b98ada3fb82632e067bf1990af71342c055b672c5d3148fad11af451"


def teacher_admission(prepared):
    fit = read(prepared["scaled_fit_ref"]["path"], prepared["scaled_fit_ref"]["sha256"])
    audit = read(prepared["scaled_fit_verification_ref"]["path"], prepared["scaled_fit_verification_ref"]["sha256"])
    require(
        fit.get("complete") is True
        and fit.get("passed") is True
        and audit.get("passed") is True
        and audit.get("phase") == "fit"
        and audit.get("fit_verified") is True,
        "028 original027 fit/audit incomplete",
    )
    require(
        fit["teacher_gate"]["passed"] is audit["teacher_gate"]["passed"] is True
        and fit["eligible_for_runtime_qualification"] is audit["eligible_for_runtime_qualification"] is True
        and fit["eligible_for_sampling"] is audit["eligible_for_sampling"] is False,
        "028 must retain027 teacher success and unopened sampling scope",
    )
    require(
        fit["prepared_ref"] == audit["prepared"] == prepared["scaled_prepared_ref"]
        and audit["input_sha256"].get(prepared["scaled_fit_ref"]["path"]) == prepared["scaled_fit_ref"]["sha256"]
        and fit["models"]["adapt"] == prepared["adapted_model_ref"],
        "028 candidate/teacher evidence identity differs",
    )
    require(
        sha256(prepared["adapted_model_ref"]["path"]) == prepared["adapted_model_ref"]["sha256"],
        "028 frozen learned coefficients or preprocessing changed",
    )
    return fit, audit


def read_prepared(path, digest):
    prepared = read(path, digest)
    require(prepared["schema_version"] == "s2a.scaled-state-preparation.v1", "028 preparation schema differs")
    require(
        prepared["config"] == read(CONFIG, CONFIG_SHA)
        and prepared["config_ref"] == reference(CONFIG)
        and prepared["registration"] == dict(path=str(PLAN), sha256=PLAN_SHA),
        "028 registered config/plan differs",
    )
    for key in ("input_sha256", "source_sha256", "snapshot_sha256"):
        check_hashes(prepared[key])
    require(
        len(prepared["source_sha256"]) == len(prepared["snapshot_sha256"]) == 96
        and prepared["source_sha256"].get(str(CONFIG)) == CONFIG_SHA
        and prepared["source_sha256"].get(str(PLAN)) == PLAN_SHA
        and str(Path(__file__).resolve()) in prepared["source_sha256"],
        "028 actual owner/registration/full source closure differs",
    )
    root = Path(prepared["output_root"])
    require(
        root == (REPO / prepared["config"]["output_dir"]).resolve() and root.is_relative_to(REPO / "outputs"),
        "028 output root differs",
    )
    weight_stats(prepared["weight_files"])
    require(
        external_identity(prepared["config"]["cosmos_repo"]) == prepared["external_cosmos"],
        "028 external tracked source changed",
    )
    teacher_admission(prepared)
    return prepared


def sampling_admission(prepared, prepared_path, precision_path, precision_sha, verification_path, verification_sha):
    precision = read(precision_path, precision_sha)
    require(
        precision.get("complete") is True
        and precision.get("passed") is True
        and precision["schema_version"] == "s2a.scaled-readout-synthetic-cpu-cuda-precision.v1"
        and precision["prepared_ref"] == reference(prepared_path)
        and precision["candidate_model_ref"] == prepared["adapted_model_ref"],
        "028 actual new-path precision admission missing or unrelated",
    )
    audit = read(verification_path, verification_sha)
    require(
        audit.get("passed") is True
        and audit.get("phase") == "precision"
        and audit.get("precision_verified") is True
        and audit.get("eligible_for_sampling") is True
        and audit["prepared"] == reference(prepared_path)
        and audit["input_sha256"].get(str(Path(precision_path).resolve())) == precision_sha,
        "028 independent precision admission missing or unrelated",
    )
    check_hashes(audit["input_sha256"])
    teacher_admission(prepared)
    return precision, audit


def owner_completed(root, stage, prepared_path):
    result = completed(root / "orchestration" / stage / "result.json", prepared_path)
    require(result["stage"] == stage and len(result["owned_commands"]) == 2, "028 owned stage budget differs")
    for item in result["owned_commands"]:
        launch, exit_record, release = (
            read(item[key]["path"], item[key]["sha256"]) for key in ("launch", "exit", "release")
        )
        require(
            exit_record == dict(returncode=0, actual_wait=True, automatic_retry=False)
            and all(release[key] == launch["child"][key] for key in ("pid", "start_ticks"))
            and release["reaped"] is True
            and release["returncode"] == 0
            and release["original_error"] is None
            and release["signal_sent"] is False,
            "028 prior command has no successful actual wait/reap",
        )
    released = read(result["gpu_after_ref"]["path"], result["gpu_after_ref"]["sha256"])
    require(
        released["released"] is True and released["memory_mib"] == 0 and not released["compute_apps"],
        "028 prior actual GPU release missing",
    )
    return result


def planned_commands(args, prepared):
    root = Path(prepared["output_root"])
    model_python = str(Path(prepared["config"]["cosmos_repo"]) / ".venv/bin/python")
    common = ["--prepared", str(args.prepared.resolve()), "--prepared-sha256", args.prepared_sha256]
    precision, precision_verified = root / "precision/result.json", root / "review/precision_verified.json"
    if args.stage == "precision":
        worker = [model_python, "scripts/probe_wam_scaled_readout_quantization.py", *common, "--seed", "42"]
        producer = precision
    else:
        worker = [
            model_python,
            "scripts/sample_wam_scaled_readout.py",
            *common,
            "--precision",
            str(precision),
            "--precision-sha256",
            sha256(precision),
            "--precision-verification",
            str(precision_verified),
            "--precision-verification-sha256",
            sha256(precision_verified),
            "--seed",
            "42",
        ]
        producer = root / "sampling/result.json"
    audit = [
        str(REPO / ".venv/bin/python"),
        "scripts/verify_wam_scaled_state.py",
        "--phase",
        args.stage,
        "--prepared",
        str(args.prepared.resolve()),
        "--precision",
        str(precision),
    ]
    if args.stage == "sampling":
        audit += ["--precision-verification", str(precision_verified), "--sampling", str(producer)]
    audit += ["--output", str(root / "review" / f"{args.stage}_verified.json"), "--seed", "42"]
    return worker, audit, producer


def execute(args):
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "028 owner requires S2A cwd and empty CUDA/LD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    peer = read(args.source_peer, args.source_peer_sha256)
    require(
        peer.get("passed") is True
        and peer.get("phase") == "prepare"
        and peer.get("preparation_verified") is True
        and peer["prepared"] == reference(args.prepared),
        "028 independent prepare admission missing or unrelated",
    )
    check_hashes(peer["input_sha256"])
    root = Path(prepared["output_root"])
    if args.stage == "sampling":
        owner_completed(root, "precision", args.prepared)
        precision, audit = root / "precision/result.json", root / "review/precision_verified.json"
        sampling_admission(prepared, args.prepared, precision, sha256(precision), audit, sha256(audit))
    worker, audit, producer = planned_commands(args, prepared)
    if args.dry_run:
        print(json.dumps(dict(stage=args.stage, worker=worker, audit=audit, physical_gpu=6, new_model_calls=0)))
        return
    directory = root / "orchestration" / args.stage
    require(
        not directory.exists() and not (root / args.stage).exists() and not producer.exists(),
        "028 refuses existing owner/producer output",
    )
    directory.mkdir(parents=True)
    result = dict(
        schema_version="s2a.scaled-state-owner.v1",
        stage=args.stage,
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        source_peer=reference(args.source_peer),
        owned_commands=[],
        automatic_retry=False,
    )
    try:
        lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks/physical6.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = resource_observation()
            write_once(directory / "gpu_before.json", before)
            require(before["memory_mib"] == 0 and not before["compute_apps"], "028 GPU6 occupied; tasks untouched")
            if args.stage == "sampling":
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", prepared["config"]["master_port"]))
            read_prepared(args.prepared, args.prepared_sha256)
            env = environment(prepared, True)
            env["MASTER_PORT"] = str(prepared["config"]["master_port"])
            command = owned_command(worker, env, directory / "model", started=root / args.stage / "worker_started.json")
            result["owned_commands"].append(command)
            result["gpu_after_ref"] = wait_release(directory)
            result["resource_observations_ref"] = reference(directory / "resource_observations.jsonl")
        raw = completed(root / args.stage / "worker_result.json", args.prepared)
        raw.update(
            exit_ref=command["exit"],
            release_ref=command["release"],
            gpu_after_ref=result["gpu_after_ref"],
            resource_observations_ref=result["resource_observations_ref"],
        )
        write_once(producer, raw)
        result["owned_commands"].append(owned_command(audit, environment(prepared, False), directory / "independent"))
        checked = verified_receipt(root / "review" / f"{args.stage}_verified.json", args.stage, args.prepared, producer)
        if args.stage == "precision":
            require(
                checked["precision_verified"] is True and checked["eligible_for_sampling"] is True,
                "028 precision rejected",
            )
        else:
            result["eligible_for_development_physics_registration"] = checked[
                "eligible_for_development_physics_registration"
            ]
            result["physics_authorized"] = False
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
    raise InterruptedError(f"028 owner interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("precision", "sampling"), required=True)
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
