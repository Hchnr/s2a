"""Own the preregistered027 CPU fit and its independent audit; no model stage."""

import argparse
import json
import os
import signal
from pathlib import Path

from prepare_wam_flow_readout import weight_stats
from run_wam_flow_readout import check_hashes, completed, environment, read, reference, require, verified
from run_wam_readout import external_identity, owned_command

from s2a.artifacts import REPO
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/wam_train_scale_readout_20261021.json"
CONFIG_SHA = "dd0e0124d256a953af48ea2e548cca6603cd9f777d6bd5e4f620548591a2c06b"
PLAN = REPO / "plans/v0.0.1/027_train_scale_readout_20261021.md"
PLAN_SHA = "586ddf2656d39752efd7979e1877af940004e14b058d900c7ef464661cbf9fb1"


def read_prepared(path, digest):
    prepared = read(path, digest)
    require(prepared["schema_version"] == "s2a.scaled-readout-preparation.v1", "027 preparation schema differs")
    require(
        prepared["config"] == read(CONFIG, CONFIG_SHA)
        and prepared["config_ref"] == reference(CONFIG)
        and prepared["registration"] == dict(path=str(PLAN), sha256=PLAN_SHA),
        "027 registered config/plan identity differs",
    )
    for key in ("input_sha256", "source_sha256", "snapshot_sha256"):
        check_hashes(prepared[key])
    require(
        prepared["source_sha256"].get(str(CONFIG)) == CONFIG_SHA
        and prepared["source_sha256"].get(str(PLAN)) == PLAN_SHA,
        "027 source closure omitted registered config/plan",
    )
    root = Path(prepared["output_root"])
    require(
        root == (REPO / prepared["config"]["output_dir"]).resolve() and root.is_relative_to(REPO / "outputs"),
        "027 output root differs",
    )
    require(str(Path(__file__).resolve()) in prepared["source_sha256"], "027 preparation omitted actual owner")
    weight_stats(prepared["weight_files"])
    require(
        external_identity(prepared["config"]["cosmos_repo"]) == prepared["external_cosmos"],
        "027 external tracked source changed",
    )
    return prepared


def planned_commands(args, prepared):
    root = Path(prepared["output_root"])
    python = str(REPO / ".venv/bin/python")
    worker = [
        python,
        "scripts/fit_wam_scaled_readout.py",
        "--prepared",
        str(args.prepared.resolve()),
        "--prepared-sha256",
        args.prepared_sha256,
        "--seed",
        "42",
    ]
    audit = [
        python,
        "scripts/verify_wam_scaled_readout.py",
        "--phase",
        "fit",
        "--prepared",
        str(args.prepared.resolve()),
        "--fit",
        str(root / "fit/fit_result.json"),
        "--output",
        str(root / "review/fit_verified.json"),
        "--seed",
        "42",
    ]
    return worker, audit


def fit_admission(producer, audit):
    require(
        producer["candidate_fits"] == producer["fit_attempts"] == producer["producer_fit_groups"] == 1
        and producer["reused_controls"] == 1
        and audit["independent_model_reconstructions"] == 1,
        "027 single-fit/reused-control budget differs",
    )
    require(
        producer["fitted_rows"] == producer["validation_rows"] == 2880 and audit["fit_verified"] is True,
        "027 complete whole-episode fit was not verified",
    )
    require(
        producer["teacher_gate"] == audit["teacher_gate"]
        and producer["eligible_for_runtime_qualification"] is audit["eligible_for_runtime_qualification"]
        and producer["eligible_for_runtime_qualification"] is producer["teacher_gate"]["passed"],
        "027 producer and independent teacher decisions differ",
    )
    require(
        producer["eligible_for_sampling"] is audit["eligible_for_sampling"] is False,
        "027 CPU evidence cannot authorize model sampling",
    )


def execute(args):
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "027 owner requires S2A cwd and empty CUDA/LD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    peer = read(args.source_peer, args.source_peer_sha256)
    require(
        peer.get("passed") is True
        and peer.get("phase") == "prepare"
        and peer.get("preparation_verified") is True
        and peer["prepared"] == reference(args.prepared),
        "027 independent prepare admission missing or unrelated",
    )
    check_hashes(peer["input_sha256"])
    worker, audit = planned_commands(args, prepared)
    if args.dry_run:
        print(json.dumps(dict(worker=worker, audit=audit, new_model_calls=0, gpu_processes=0, candidate_fits=0)))
        return
    root = Path(prepared["output_root"])
    directory = root / "orchestration/fit"
    require(
        not directory.exists() and not (root / "fit").exists() and not (root / "review/fit_verified.json").exists(),
        "027 refuses existing owner, fit or independent output",
    )
    directory.mkdir(parents=True)
    result = dict(
        schema_version="s2a.scaled-readout-owner.v1",
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        source_peer=reference(args.source_peer),
        owned_commands=[],
        automatic_retry=False,
        new_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    try:
        result["owned_commands"].append(owned_command(worker, environment(prepared, False), directory / "producer"))
        producer_path = root / "fit/fit_result.json"
        producer = completed(producer_path, args.prepared)
        result["owned_commands"].append(owned_command(audit, environment(prepared, False), directory / "independent"))
        checked = verified(root / "review/fit_verified.json", "fit", args.prepared, producer_path)
        fit_admission(producer, checked)
        read_prepared(args.prepared, args.prepared_sha256)
        check_hashes(peer["input_sha256"])
        result.update(
            complete=True,
            passed=True,
            producer_result=reference(producer_path),
            eligible_for_runtime_qualification=checked["eligible_for_runtime_qualification"],
            eligible_for_sampling=False,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_once(directory / "result.json", result)
    print(json.dumps(dict(result=reference(directory / "result.json"))))


def interrupted(signum, frame):
    raise InterruptedError(f"027 owner interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
