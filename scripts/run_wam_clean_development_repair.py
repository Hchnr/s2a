"""One051 owner for the fixed twelve-case development continuation."""

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

from run_wam_clean_training_continuation import observe_descendant_cleanup

from s2a.clean_training_d1_admission import Evidence
from s2a.clean_training_development_handoff import environment as original_environment
from s2a.clean_training_w1_handoff import PREPARED_ROOT
from s2a.development_repair_admission import BUNDLE, ENV_PATH, ENV_SHA, PREFIX, ROOT, admit
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

NAMESPACE = "development_dependency_repair_20261024"
ENTRY = "scripts/run_wam_clean_development_repair.py"


def command(a):
    argv = [str(REPO / ".venv/bin/python"), str(BUNDLE / "run_wam_clean_training_development.py")]
    for name, key in (
        ("prepared", "prepared_ref"),
        ("execution-registration", "execution_registration_ref"),
        ("state-owner", "state_owner_ref"),
        ("state-verification", "state_verification_ref"),
    ):
        argv += ["--" + name, a[key]["path"], "--" + name + "-sha256", a[key]["sha256"]]
    return argv + ["--seed", "42"]


def environment(ref):
    env = original_environment()
    env[ENV_PATH] = ref["path"]
    env[ENV_SHA] = ref["sha256"]
    env["PYTHONPATH"] = os.pathsep.join([str(BUNDLE), env["PYTHONPATH"]])
    return env


def absent():
    for p in (
        ROOT / "supervisor",
        ROOT / "role_receipts",
        PREPARED_ROOT / NAMESPACE,
        PREPARED_ROOT / "orchestration/DEVELOPMENT_dependency_repair_20261024",
        PREPARED_ROOT / "review/DEVELOPMENT_dependency_repair_20261024_verified.json",
    ):
        require(not p.exists(), "051 actual outputs exist; no retry: " + str(p))


def check_result(a, ref):
    e = Evidence()
    owner_ref = reference(PREPARED_ROOT / "orchestration/DEVELOPMENT_dependency_repair_20261024/result.json")
    owner = e.json(owner_ref)
    require(
        owner["schema_version"] == "s2a.clean-training-development-audit-owner.v1"
        and owner["complete"] is owner["passed"] is True,
        "051 original development owner incomplete",
    )
    audit = e.json(owner["verification_ref"])
    run = e.json(owner["development_ref"])
    require(
        audit["schema_version"] == "s2a.clean-training-development-verification.v1"
        and audit["complete"] is audit["passed"] is True
        and run["complete"] is run["passed"] is True
        and audit["development_ref"] == owner["development_ref"],
        "051 independent full audit missing",
    )
    for key in ("prepared_ref", "execution_registration_ref"):
        require(owner[key] == run[key] == audit[key] == a[key], "051 original041 lineage differs")
    for key in ("state_owner_ref", "state_verification_ref"):
        require(owner[key] == run[key] == a[key], "051 STATE lineage differs")
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
        and len(audit["cases"]) == len(run["owned_commands"]) == 12
        and owner["score"] == audit["score"],
        "051 full fixed budget/score differs",
    )
    require(
        len(owner["owned_commands"]) == 1
        and owner["automatic_retry"] is owner["fresh_performance_authorized"] is False,
        "051 audit ownership differs",
    )
    role_commands = [
        ("model", owner["service_owned_command"]),
        ("audit", owner["owned_commands"][0]),
        *[(f"client_{i:02d}", c) for i, c in enumerate(run["owned_commands"])],
    ]
    for role, entry in role_commands:
        launch = normal_receipts(e, entry["launch"], entry["exit"], entry["release"], format="legacy")
        receipt = e.json(reference(ROOT / "role_receipts" / (role + ".json")))
        require(
            receipt["registration_ref"] == ref
            and all(receipt["identity"][k] == launch["child"][k] for k in ("pid", "start_ticks")),
            "051 actual role birth/registration differs",
        )
    gpu = e.json(owner["gpu_after_ref"])
    require(
        gpu["released"] is True
        and gpu["uuid"] == a["gpu_uuid"]
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == [],
        "051 GPU not released",
    )
    e.mapping(a["source_sha256"])
    e.mapping(a["snapshot_sha256"])
    e.finish()
    return owner_ref


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


def execute(ref, *, mode):
    check_cpu_environment(os.environ)
    require(Path.cwd().resolve() == REPO, "051 cwd differs")
    a = admit(ref)
    sources = a["source_sha256"]
    if mode == "dry-run":
        result = subprocess.run(
            command(a) + ["--dry-run"], cwd=REPO, env=environment(ref), capture_output=True, text=True, timeout=180
        )
        require(result.returncode == 0, "051 original CLI dry failed: " + result.stderr[-2500:])
        actual = json.loads(result.stdout)
        old = read(a["old049_ref"])
        require(
            actual["case_order"] == old["case_order"]
            and actual["actual_model_gpu_uuid"] == a["gpu_uuid"]
            and actual["model_initializations"] == actual["new_physics_steps"] == 0,
            "051 dry matrix/budget differs",
        )
        check_sources(sources)
        return dict(
            complete=True, passed=True, actual_original_dry=actual, actual_model_calls=0, actual_physics_steps=0
        )
    directory = ROOT / "supervisor"
    if mode == "launch":
        absent()
        return launch_detached(
            argv(ref, "--run"), directory=directory, source_sha256=sources, environment=cpu_environment()
        )
    require(mode == "run", "051 mode differs")
    launch = accept_detached_launch(directory, command=argv(ref, "--run"), source_sha256=sources)
    result = dict(
        schema_version=PREFIX + "owner.v1",
        complete=False,
        passed=False,
        registration_ref=ref,
        launch_ref=launch,
        identity=identity(os.getpid()),
        automatic_retry=False,
        fresh_performance_authorized=False,
    )

    def interrupted(signum, frame):
        raise InterruptedError(f"051 interrupted by {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        owned = owned_stage(
            command(a),
            environment=environment(ref),
            directory=directory / "DEVELOPMENT",
            timeout_s=172980,
            source_sha256=sources,
            cleanup_grace_s=130,
            observe_cleanup=lambda: observe_descendant_cleanup(PREPARED_ROOT),
            check_result=lambda: check_result(a, ref),
        )
        owner_ref = check_result(a, ref)
        owner = read(owner_ref)
        birth = read(reference(ROOT / "role_receipts/owner.json"))["identity"]
        child = read(read(owned)["launch_ref"])["child"]
        require(
            all(birth[k] == child[k] for k in ("pid", "start_ticks", "boot_id")),
            "051 owner source receipt birth differs",
        )
        check_sources(sources)
        check_sources(a["snapshot_sha256"])
        result.update(
            complete=True,
            passed=True,
            owned_development_ref=owned,
            development_owner_ref=owner_ref,
            score=owner["score"],
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
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
    for mode in ("dry-run", "launch", "run"):
        group.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    mode = "dry-run" if args.dry_run else "launch" if args.launch else "run"
    print(json.dumps(execute(dict(path=str(args.registration.resolve()), sha256=args.registration_sha256), mode=mode)))


if __name__ == "__main__":
    main()
