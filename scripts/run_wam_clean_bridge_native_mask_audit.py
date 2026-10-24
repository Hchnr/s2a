"""Own one CPU audit of saved 037 captures under the registered native mask."""

import argparse
import json
import os
import shutil
import signal
from pathlib import Path

import run_wam_clean_bridge as bridge
from run_wam_readout import owned_command

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/wam_clean_bridge_native_mask_audit_20261023.json"
PLAN = REPO / "plans/v0.0.1/039_native_mask_audit_addendum_20261023.md"
NEW_BUDGET = dict(model_initializations=0, denoise_calls=0, fit_groups=0, physics_steps=0, fresh_cases=0)


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def read(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "039 exact reference required")
    require(reference(ref["path"]) == ref, "039 evidence reference changed")
    return json.loads(Path(ref["path"]).read_text())


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_once(path, value)
    return reference(path)


def prior_evidence(config):
    prepared = bridge.read_prepared(config["prepared_ref"])
    original_root = Path(prepared["root"])
    qualification = read(config["qualification_ref"])
    original_owner = read(config["original_owner_ref"])
    require(
        config["qualification_ref"] == reference(original_root / "qualification/result.json")
        and config["original_owner_ref"] == reference(original_root / "orchestration/result.json"),
        "039 references must belong to the same complete model run",
    )
    require(
        qualification["passed"] is qualification["complete"] is qualification["bridge_qualified"] is True
        and qualification["prepared_ref"] == config["prepared_ref"]
        and qualification["actual_requests"] == 7
        and qualification["actual_denoise_calls"] == qualification["actual_head_calls"] == 245
        and all(
            qualification[key] == 0
            for key in ("new_candidate_fits", "new_episode_reads", "new_future_reads", "physics_steps", "fresh_cases")
        ),
        "039 requires complete seven-request model evidence without fitting or future reads",
    )
    require(
        original_owner["complete"] is original_owner["passed"] is original_owner["bridge_qualified"] is False
        and original_owner["prepared_ref"] == config["prepared_ref"]
        and original_owner["error"]["type"] == "ChildProcessError"
        and len(original_owner["owned_commands"]) == 1,
        "039 must preserve the original failed owner and complete model child",
    )
    bridge.check_owned_command(original_owner["owned_commands"][0])
    release = read(config["gpu_after_ref"])
    require(
        config["gpu_after_ref"] == reference(original_root / "orchestration/gpu_after.json")
        and original_owner["gpu_after_ref"] == config["gpu_after_ref"]
        and release["released"] is True
        and release["memory_mib"] == 0
        and release["compute_apps"] == []
        and release["uuid"] == prepared["config"]["gpu_uuid"],
        "039 original model GPU release is missing",
    )
    failed = original_root / "orchestration/independent"
    failed_exit = read(config["original_checker_exit_ref"])
    failed_release = read(config["original_checker_release_ref"])
    require(
        config["original_checker_exit_ref"] == reference(failed / "exit.json")
        and config["original_checker_release_ref"] == reference(failed / "ownership/release.json")
        and failed_exit == dict(returncode=1, actual_wait=True, automatic_retry=False)
        and failed_release["returncode"] == 1
        and failed_release["reaped"] is True
        and failed_release["signal_sent"] is False,
        "039 prior failed checker must have its actual exit and cleanup",
    )
    log = config["original_checker_log_ref"]
    require(
        reference(log["path"]) == log and Path(log["path"]) == failed / "command.log", "039 old failure log changed"
    )
    require(
        "ValueError: initial sample contains privileged action/future conditioning" in Path(log["path"]).read_text(),
        "039 correction does not describe the recorded checker failure",
    )
    diagnosis = config["diagnosis_ref"]
    require(reference(diagnosis["path"]) == diagnosis, "039 registered mask diagnosis changed")
    return prepared, original_owner


def execute(config_path):
    bridge.cpu_owner()
    config_ref = reference(config_path)
    config = read(config_ref)
    require(
        config["seed"] == 42
        and config["new_budget"] == NEW_BUDGET
        and all(
            config[key] is False
            for key in ("automatic_retry", "model_training_authorized", "fresh_performance_authorized")
        ),
        "039 budget or authorization changed",
    )
    root = (REPO / config["output_dir"]).resolve()
    require(root.is_relative_to(REPO / "outputs") and not root.exists(), "039 requires a new local output root")
    prepared, original_owner = prior_evidence(config)
    require(root != Path(prepared["root"]), "039 cannot write to the original run")
    source_files = [
        Path(__file__).resolve(),
        REPO / "scripts/verify_wam_clean_bridge_native_mask.py",
        REPO / "tests/test_wam_clean_bridge_native_mask.py",
        Path(config_path).resolve(),
        PLAN,
    ]
    sources = dict(prepared["source_sha256"])
    for path in source_files:
        require(path.is_file() and path.is_relative_to(REPO), "039 source must be an existing S2A file")
        require(str(path) not in sources, "039 cannot replace a frozen 037 source")
        sources[str(path)] = sha256(path)
    root.mkdir(parents=True)
    snapshots = {}
    for path in source_files:
        target = root / "source/s2a" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        snapshots[str(target)] = sha256(target)
        require(snapshots[str(target)] == sources[str(path)], "039 source changed while copied")
    registration_ref = publish(
        root / "preparation/prepared.json",
        dict(
            schema_version="s2a.clean-bridge-native-mask-audit-preparation.v1",
            config_ref=config_ref,
            plan_ref=reference(PLAN),
            original_prepared_ref=config["prepared_ref"],
            qualification_ref=config["qualification_ref"],
            original_failed_owner_ref=config["original_owner_ref"],
            diagnosis_ref=config["diagnosis_ref"],
            source_sha256=sources,
            new_snapshot_sha256=snapshots,
            new_budget=NEW_BUDGET,
            output_routing="validated original manifest; in-memory root changes output path only",
        ),
    )
    directory = root / "orchestration"
    directory.mkdir()
    result = dict(
        schema_version="s2a.clean-bridge-native-mask-audit-owner.v1",
        complete=False,
        passed=False,
        bridge_qualified=False,
        registration_ref=registration_ref,
        original_prepared_ref=config["prepared_ref"],
        qualification_ref=config["qualification_ref"],
        original_failed_owner_ref=config["original_owner_ref"],
        preserved_original_owner_passed=original_owner["passed"],
        new_budget=NEW_BUDGET,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        automatic_retry=False,
        owned_commands=[],
    )
    try:
        command = [
            str(REPO / ".venv/bin/python"),
            "scripts/verify_wam_clean_bridge_native_mask.py",
            "--prepared",
            config["prepared_ref"]["path"],
            "--prepared-sha256",
            config["prepared_ref"]["sha256"],
            "--qualification",
            config["qualification_ref"]["path"],
            "--output",
            str(root / "review/verified.json"),
            "--seed",
            "42",
        ]
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            LD_LIBRARY_PATH="",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(REPO / "src") + ":" + str(REPO / "scripts"),
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
        )
        bridge.check_hashes(sources)
        bridge.check_hashes(snapshots)
        prior_evidence(config)
        print(json.dumps(dict(stage="independent_native_mask_audit", new_model_calls=0)), flush=True)
        entry = owned_command(command, environment, directory / "independent", timeout=1800)
        result["owned_commands"].append(entry)
        bridge.check_owned_command(entry)
        verified_ref = reference(root / "review/verified.json")
        verified = read(verified_ref)
        require(
            verified["passed"] is verified["complete"] is verified["bridge_qualified"] is True
            and verified["prepared_ref"] == config["prepared_ref"]
            and verified["qualification_ref"] == config["qualification_ref"]
            and verified["actual_online_points_verified"] == 245
            and verified["model_training_authorized"] is verified["fresh_performance_authorized"] is False,
            "039 complete independent bridge qualification did not pass",
        )
        prior_evidence(config)
        bridge.check_hashes(sources)
        bridge.check_hashes(snapshots)
        result.update(complete=True, passed=True, bridge_qualified=True, review_ref=verified_ref)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(directory / "result.json", result)
    print(json.dumps(reference(directory / "result.json")), flush=True)


def interrupted(signum, frame):
    raise InterruptedError(f"039 CPU owner interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    execute(args.config.resolve())


if __name__ == "__main__":
    main()
