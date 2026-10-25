"""Own one original twelve-case development run after full joint048 admission."""

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

from run_wam_clean_training_continuation import observe_descendant_cleanup

from s2a.clean_training_d1_admission import Evidence
from s2a.clean_training_development_handoff import (
    PREFIX,
    PREPARED_ROOT,
    ROOT,
    absent_outputs,
    admit,
    command,
    environment,
)
from s2a.durable_stage_supervisor import (
    REPO,
    accept_detached_launch,
    check_cpu_environment,
    check_sources,
    cpu_environment,
    identity,
    launch_detached,
    owned_stage,
    publish,
    read,
    reference,
    require,
)
from s2a.owned_receipt_contract import normal_receipts

ENTRY = "scripts/run_wam_clean_training_development_handoff.py"


def argv(ref, mode):
    return [
        str(REPO / ".venv/bin/python"),
        ENTRY,
        "--registration",
        ref["path"],
        "--registration-sha256",
        ref["sha256"],
        "--seed",
        "42",
        mode,
    ]


def check_result(a):
    e = Evidence()
    owner_ref = reference(PREPARED_ROOT / "orchestration/DEVELOPMENT/result.json")
    owner = e.json(owner_ref)
    require(
        owner["schema_version"] == "s2a.clean-training-development-audit-owner.v1"
        and owner["complete"] is owner["passed"] is True
        and owner["automatic_retry"] is owner["fresh_performance_authorized"] is False,
        "original development audit owner incomplete",
    )
    audit = e.json(owner["verification_ref"])
    run = e.json(owner["development_ref"])
    require(
        owner["verification_ref"] == reference(PREPARED_ROOT / "review/DEVELOPMENT_verified.json")
        and audit["schema_version"] == "s2a.clean-training-development-verification.v1"
        and audit["complete"] is audit["passed"] is True
        and run["schema_version"] == "s2a.clean-training-development-owner.v1"
        and run["complete"] is run["passed"] is True
        and audit["development_ref"] == owner["development_ref"],
        "original development independent audit differs",
    )
    for key in ("prepared_ref", "execution_registration_ref"):
        require(owner[key] == audit[key] == run[key] == a[key], "development original041 identity differs")
    for key in ("state_owner_ref", "state_verification_ref"):
        require(owner[key] == run[key] == a[key], "development STATE lineage differs")
    require(
        all(
            audit[key] == value
            for key, value in dict(
                actual_requests=900,
                actual_denoise_calls=31500,
                actual_head_calls=31500,
                physics_steps=7200,
                model_initializations=1,
                paired_first_current_conditions=6,
                independently_verified_compact_points=31500,
                future_label_archive_decodes=0,
                fresh_cases=0,
                consumed_before_after_exact=True,
            ).items()
        )
        and len(audit["cases"]) == 12
        and owner["score"] == audit["score"],
        "full twelve-case independent qualification missing",
    )
    require(len(owner["owned_commands"]) == 1, "original development lacks independent audit child")
    for item in [owner["service_owned_command"], *owner["owned_commands"], *run["owned_commands"]]:
        normal_receipts(e, item["launch"], item["exit"], item["release"], format="legacy")
    gpu = e.json(owner["gpu_after_ref"])
    require(
        gpu["released"] is True
        and gpu["uuid"] == a["gpu_uuid"]
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == [],
        "development GPU release missing",
    )
    e.finish()
    return owner_ref


def interrupted(signum, frame):
    raise InterruptedError(f"049 supervisor interrupted by signal {signum}")


def execute(ref, *, mode):
    check_cpu_environment(os.environ)
    require(Path.cwd().resolve() == REPO, "049 must run from S2A")
    a = admit(ref)
    sources = a["source_sha256"]
    if mode == "dry-run":
        actual = subprocess.run(
            [*command(a), "--dry-run"], cwd=REPO, env=environment(), capture_output=True, text=True, timeout=120
        )
        require(actual.returncode == 0, "original development dry failed: " + actual.stderr[-1500:])
        original = json.loads(actual.stdout)
        require(
            original["case_order"] == a["case_order"]
            and original["actual_model_gpu_uuid"] == a["gpu_uuid"]
            and original["model_initializations"] == original["new_physics_steps"] == 0,
            "original development dry order, UUID or budget differs",
        )
        check_sources(sources)
        check_sources(a["snapshot_sha256"])
        return dict(
            original_dry=original,
            schema_version=PREFIX + "dry.v1",
            complete=True,
            passed=True,
            registration_ref=ref,
            command=[*command(a), "--dry-run"],
            budget=a["budget"],
            actual_model_calls=0,
            actual_physics_steps=0,
        )
    directory = ROOT / "supervisor"
    if mode == "launch":
        absent_outputs()
        return launch_detached(
            argv(ref, "--run"), directory=directory, source_sha256=sources, environment=cpu_environment()
        )
    require(mode == "run", "049 mode differs")
    launch = accept_detached_launch(directory, command=argv(ref, "--run"), source_sha256=sources)
    result = dict(
        schema_version=PREFIX + "owner.v1",
        complete=False,
        passed=False,
        registration_ref=ref,
        identity=identity(os.getpid()),
        launch_ref=launch,
        automatic_retry=False,
        fresh_performance_authorized=False,
    )
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        owned = owned_stage(
            command(a),
            environment=environment(),
            directory=directory / "DEVELOPMENT",
            timeout_s=a["owner_timeout_s"],
            source_sha256=sources,
            cleanup_grace_s=130,
            observe_cleanup=lambda: observe_descendant_cleanup(PREPARED_ROOT),
            check_result=lambda: check_result(a),
        )
        owner_ref = check_result(a)
        check_sources(sources)
        check_sources(a["snapshot_sha256"])
        result.update(
            complete=True,
            passed=True,
            owned_original_owner_ref=owned,
            development_owner_ref=owner_ref,
            score=read(owner_ref)["score"],
        )
    except BaseException as error:
        result.update(complete=False, passed=False, error=dict(type=type(error).__name__, message=str(error)))
        failed = directory / "DEVELOPMENT/result.json"
        if failed.exists():
            result["failed_owned_original_owner_ref"] = reference(failed)
        raise
    finally:
        try:
            publish(directory / "result.json", result)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return reference(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "launch", "run"):
        group.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    mode = "dry-run" if args.dry_run else "launch" if args.launch else "run"
    print(json.dumps(execute(dict(path=str(args.registration.resolve()), sha256=args.registration_sha256), mode=mode)))


if __name__ == "__main__":
    main()
