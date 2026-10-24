"""Launch/own the single041 recovery pipeline independently of tool sessions."""

import argparse
import fcntl
import json
import os
import signal
import uuid
from pathlib import Path

from s2a.clean_execution_proof import check_execution_proof
from s2a.durable_stage_supervisor import (
    REPO,
    accept_detached_launch,
    check_cpu_environment,
    check_sources,
    checked,
    cpu_environment,
    identity,
    launch_detached,
    owned_stage,
    publish,
    read,
    reference,
    require,
    sha,
)

NEW_ROOT_REL = Path("outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023")
STAGES = ("D0", "W1", "D1", "W2", "STATE", "DEVELOPMENT")
TIMEOUTS = dict(D0=88400, W1=15000, D1=88400, W2=15000, STATE=45500, DEVELOPMENT=175000)


def expected_supervisor():
    def entry(stage):
        basename = (
            "run_wam_clean_training_state.py"
            if stage == "STATE"
            else "run_wam_clean_training_development.py"
            if stage == "DEVELOPMENT"
            else "run_wam_clean_training.py"
        )
        return dict(
            entrypoint=str(NEW_ROOT_REL / "source/entrypoints" / basename),
            arguments=[] if stage in ("STATE", "DEVELOPMENT") else ["--stage", stage],
        )

    return dict(
        python=str(REPO / ".venv/bin/python"),
        stage_order=list(STAGES),
        stage_commands={stage: entry(stage) for stage in STAGES},
        stage_timeouts_s=dict(TIMEOUTS),
        result_paths={stage: f"orchestration/{stage}/result.json" for stage in STAGES},
    )


def admission(prepared_ref, registration_ref):
    require(Path.cwd().resolve() == REPO, "041 supervisor must run from S2A")
    check_cpu_environment(os.environ)
    prepared, registration = read(prepared_ref), read(registration_ref)
    require(
        prepared.get("schema_version") == "s2a.clean-training-preparation.v1"
        and Path(prepared["root"]).resolve() == REPO / NEW_ROOT_REL,
        "041 preparation root/schema differs",
    )
    require(
        registration.get("schema_version") == "s2a.clean-training-execution-registration.v1"
        and registration.get("prepared_ref") == prepared_ref
        and registration.get("complete") is True
        and registration.get("passed") is True
        and registration.get("execution_authorized") is True
        and registration.get("readiness", {}).get("ready_for_execution") is True
        and registration.get("automatic_retry") is False
        and registration.get("fresh_performance_authorized") is False,
        "041 requires an independent complete execution envelope",
    )
    check_execution_proof(registration, prepared_ref=prepared_ref, read_json=read, check_file=checked)
    contract = registration.get("continuation_contract", {})
    require(
        set(contract)
        == {
            "protocol",
            "original_execution_ref",
            "incident_ref",
            "old_data_used_for_fit",
            "one_new_complete_attempt",
            "automatic_retry",
        }
        and contract["protocol"] == "041"
        and contract["old_data_used_for_fit"] is False
        and contract["one_new_complete_attempt"] is True
        and contract["automatic_retry"] is False,
        "041 continuation contract differs",
    )
    checked(contract["original_execution_ref"])
    checked(contract["incident_ref"])
    require(registration.get("supervisor") == expected_supervisor(), "041 exact six-stage supervisor plan differs")
    metadata = read(registration["metadata_verification_ref"])
    require(
        metadata.get("schema_version") == "s2a.clean-training-preparation-independent.v1"
        and metadata.get("prepared_ref") == prepared_ref
        and metadata.get("complete") is True
        and metadata.get("passed") is True
        and metadata.get("metadata_qualified") is True
        and metadata.get("execution_ready") is False
        and metadata.get("execution_authorized") is False,
        "041 metadata qualification is missing or unrelated",
    )
    sources = registration["source_sha256"]
    check_sources(sources)
    check_sources(registration["snapshot_sha256"])
    for filename in (
        Path(__file__),
        REPO / "src/s2a/durable_stage_supervisor.py",
        Path(metadata["verifier_ref"]["path"]),
        *(REPO / entry["entrypoint"] for entry in registration["supervisor"]["stage_commands"].values()),
    ):
        absolute = str(filename.resolve())
        require(sources.get(absolute) == sha(absolute), "041 supervisor/owner/verifier source is not execution-frozen")
    checked(metadata["verifier_ref"])
    require(
        registration.get("root") == prepared["root"],
        "041 execution and preparation roots differ",
    )
    return prepared, registration


def build_stage_argv(registration, prepared_ref, registration_ref, stage):
    require(stage in STAGES, "unregistered041 stage")
    spec = registration["supervisor"]["stage_commands"][stage]
    return [
        registration["supervisor"]["python"],
        spec["entrypoint"],
        "--prepared",
        prepared_ref["path"],
        "--prepared-sha256",
        prepared_ref["sha256"],
        "--execution-registration",
        registration_ref["path"],
        "--execution-registration-sha256",
        registration_ref["sha256"],
        "--seed",
        "42",
        *spec["arguments"],
    ]


def run_argv(prepared_ref, registration_ref):
    return [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_clean_training_continuation.py",
        "--prepared",
        prepared_ref["path"],
        "--prepared-sha256",
        prepared_ref["sha256"],
        "--execution-registration",
        registration_ref["path"],
        "--execution-registration-sha256",
        registration_ref["sha256"],
        "--seed",
        "42",
        "--run",
    ]


def check_owner_result(root, registration, prepared_ref, registration_ref, stage):
    path = root / registration["supervisor"]["result_paths"][stage]
    result_ref = reference(path)
    value = read(result_ref)
    schema = (
        "s2a.clean-training-state-owner.v1"
        if stage == "STATE"
        else "s2a.clean-training-development-audit-owner.v1"
        if stage == "DEVELOPMENT"
        else "s2a.clean-training-owner.v1"
    )
    require(
        value.get("schema_version") == schema
        and value.get("complete") is True
        and value.get("passed") is True
        and value.get("prepared_ref") == prepared_ref
        and value.get("execution_registration_ref") == registration_ref
        and (stage == "DEVELOPMENT" or value.get("stage") == stage),
        "041 stage owner result incomplete or unrelated",
    )
    # DEVELOPMENT task/stability/benefit gates remain in its independent score.
    # Mechanical owner completeness must not be replaced by a score threshold.
    return result_ref


def observe_descendant_cleanup(root):
    """Inspect all recorded descendants without signalling or inventing waits."""
    entries = []
    for folder in (root / "orchestration", root / "development"):
        for pending in sorted(folder.rglob("ownership/pending.json")):
            item = dict(pending_ref=reference(pending), verified_reaped=False)
            try:
                birth = read(item["pending_ref"])
                release = pending.with_name("release.json")
                item["pid"] = birth["pid"]
                item["start_ticks"] = birth["start_ticks"]
                try:
                    actual = identity(birth["pid"])
                    item["same_birth_present"] = actual["start_ticks"] == birth["start_ticks"]
                    if item["same_birth_present"]:
                        item["current_identity"] = actual
                except (FileNotFoundError, ProcessLookupError):
                    item["same_birth_present"] = False
                if release.exists():
                    item["release_ref"] = reference(release)
                    value = read(item["release_ref"])
                    item["verified_reaped"] = (
                        value.get("reaped") is True
                        and value.get("pid") == birth["pid"]
                        and value.get("start_ticks") == birth["start_ticks"]
                        and item["same_birth_present"] is False
                    )
                else:
                    item["error"] = "missing real descendant release receipt; disappearance does not prove reap"
            except (ValueError, KeyError, OSError) as error:
                item["error"] = str(error)
            entries.append(item)
    return dict(
        schema_version="s2a.clean-training-descendant-cleanup-observation.v1",
        records=entries,
        recorded_descendants=len(entries),
        all_recorded_descendants_reaped=all(item["verified_reaped"] for item in entries),
        no_recorded_descendants_proves_no_unrecorded_descendants=False,
        observer_signals=0,
    )


def interrupted(signum, frame):
    raise InterruptedError(f"041 durable supervisor interrupted by signal{signum}")


def execute(prepared_ref, registration_ref, *, launch):
    prepared, registration = admission(prepared_ref, registration_ref)
    root = Path(prepared["root"])
    directory = root / "supervisor"
    command = run_argv(prepared_ref, registration_ref)
    sources = registration["source_sha256"]
    if launch:
        return launch_detached(command, directory=directory, source_sha256=sources, environment=cpu_environment())
    result = dict(
        schema_version="s2a.clean-training-continuation-supervisor.v1",
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        owned_commands=[],
        automatic_retry=False,
        fresh_performance_authorized=False,
        stage_order=list(STAGES),
        identity=identity(os.getpid()),
    )
    # A rejected concurrent/direct run must never publish the active owner's
    # canonical result.json. Only the process that acquired this launch/birth
    # admission may own that name.
    with (directory / "supervisor.lock").open("a") as lock:
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            accept_detached_launch(directory, command=command, source_sha256=sources)
        except BaseException as error:
            publish(
                directory / "rejected_runs" / f"{uuid.uuid4().hex}.json",
                dict(complete=True, passed=False, error=dict(type=type(error).__name__, message=str(error))),
            )
            raise
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, interrupted)
            for stage in STAGES:
                result["active_stage"] = stage
                admission(prepared_ref, registration_ref)
                stage_command = build_stage_argv(registration, prepared_ref, registration_ref, stage)
                owner_ref = owned_stage(
                    stage_command,
                    environment=cpu_environment(),
                    directory=directory / stage,
                    timeout_s=registration["supervisor"]["stage_timeouts_s"][stage],
                    source_sha256=sources,
                    cleanup_grace_s=130,
                    observe_cleanup=lambda: observe_descendant_cleanup(root),
                    check_result=lambda stage=stage: check_owner_result(
                        root, registration, prepared_ref, registration_ref, stage
                    ),
                )
                result["owned_commands"].append(dict(stage=stage, owner_ref=owner_ref))
                result["active_stage"] = None
                print(json.dumps(dict(completed_stage=stage, owner_ref=owner_ref)), flush=True)
            check_sources(sources)
            result.update(complete=True, passed=True)
        except BaseException as error:
            result["error"] = dict(type=type(error).__name__, message=str(error))
            active = result.get("active_stage")
            if active is not None and (directory / active / "result.json").is_file():
                result["failed_stage_ref"] = reference(directory / active / "result.json")
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            publish(directory / "result.json", result)
    require(result["passed"], "041 pipeline stopped; no automatic restart")
    return reference(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution-registration"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            execute(
                dict(path=str(args.prepared.resolve()), sha256=args.prepared_sha256),
                dict(path=str(args.execution_registration.resolve()), sha256=args.execution_registration_sha256),
                launch=args.launch,
            )
        )
    )


if __name__ == "__main__":
    main()
