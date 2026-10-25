"""Durably own046's one D1 wrapper and the original model/audit descendants."""

import argparse
import json
import os
import signal
from pathlib import Path

from run_wam_clean_training_continuation import observe_descendant_cleanup

from s2a.clean_training_d1_admission import PREFIX, absent_outputs, admit
from s2a.clean_training_w1_handoff import PREPARED_ROOT
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

ENTRY = "scripts/run_wam_clean_training_d1_handoff.py"
WORKER = "scripts/execute_wam_clean_training_d1_handoff.py"


def argv(registration_ref, *mode):
    return [
        str(REPO / ".venv/bin/python"),
        ENTRY,
        "--registration",
        registration_ref["path"],
        "--registration-sha256",
        registration_ref["sha256"],
        "--seed",
        "42",
        *mode,
    ]


def worker_argv(registration_ref):
    return [
        str(REPO / ".venv/bin/python"),
        WORKER,
        "--registration",
        registration_ref["path"],
        "--registration-sha256",
        registration_ref["sha256"],
        "--seed",
        "42",
    ]


def check_worker(registration, registration_ref):
    from execute_wam_clean_training_d1_handoff import check_stage

    ref = reference(Path(registration["root"]) / "worker/result.json")
    value = read(ref)
    require(
        value.get("schema_version") == PREFIX + "worker.v1"
        and value.get("complete") is value.get("passed") is True
        and value["registration_ref"] == registration_ref
        and value["w1_owner_ref"] == registration["w1_owner_ref"]
        and value["w1_params_ref"] == registration["w1_params_ref"]
        and value["delegate_calls_attempted"] == 1
        and value["source_before_after_exact"] is True,
        "046 worker did not complete its unique admitted D1",
    )
    handoff = value["handoff"]
    require(
        handoff["w1_lookups"] == 1 and handoff["restored"] is handoff["source_guard_completed"] is True,
        "046 handoff proof incomplete",
    )
    original = check_stage(registration, value["delegate_result_ref"])
    require(
        value["collection_ref"] == original["producer_ref"]
        and value["verification_ref"] == original["verification_ref"],
        "046 wrapper artifacts differ",
    )
    require(len(original["owned_commands"]) == 2, "D1 lacks model and independent audit owners")
    for command in original["owned_commands"]:
        launch, exited, released = (read(command[k]) for k in ("launch", "exit", "release"))
        require(
            exited["actual_wait"] is True
            and exited["returncode"] == 0
            and released["actual_wait"] is released["reaped"] is True
            and released["returncode"] == 0
            and released["signal_sent"] is False
            and released["original_error"] is None
            and all(released[k] == launch["child"][k] for k in ("pid", "start_ticks")),
            "D1 descendant lacks actual normally reaped birth",
        )
    gpu = read(original["gpu_after_ref"])
    require(
        gpu["released"] is True and gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "D1 GPU release not observed"
    )
    return ref


def interrupted(signum, frame):
    raise InterruptedError(f"046 supervisor interrupted by signal {signum}")


def execute(registration_ref, *, mode):
    check_cpu_environment(os.environ)
    require(Path.cwd().resolve() == REPO, "046 supervisor must run from S2A")
    registration = admit(registration_ref)
    sources = registration["source_sha256"]
    for relative in (ENTRY, WORKER):
        require(
            sources.get(str(REPO / relative)) == reference(REPO / relative)["sha256"],
            "actual046 owner/worker source not frozen",
        )
    root = Path(registration["root"])
    directory = root / "supervisor"
    if mode == "dry-run":
        return dict(
            schema_version=PREFIX + "owner-dry.v1",
            complete=True,
            passed=True,
            registration_ref=registration_ref,
            command=worker_argv(registration_ref),
            timeout_s=registration["owner_timeout_s"],
            cleanup_grace_s=130,
            actual_model_calls=0,
            actual_worker_launches=0,
        )
    if mode == "launch":
        absent_outputs()
        return launch_detached(
            argv(registration_ref, "--run"), directory=directory, source_sha256=sources, environment=cpu_environment()
        )
    require(mode == "run", "046 supervisor mode differs")
    launch = accept_detached_launch(directory, command=argv(registration_ref, "--run"), source_sha256=sources)
    result = dict(
        schema_version=PREFIX + "owner.v1",
        complete=False,
        passed=False,
        registration_ref=registration_ref,
        launch_ref=launch,
        identity=identity(os.getpid()),
        stage="D1",
        w1_owner_ref=registration["w1_owner_ref"],
        w1_params_ref=registration["w1_params_ref"],
        automatic_retry=False,
        w2_authorized=False,
        fresh_performance_authorized=False,
    )
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        owned = owned_stage(
            worker_argv(registration_ref),
            environment=cpu_environment(),
            directory=directory / "D1",
            timeout_s=registration["owner_timeout_s"],
            source_sha256=sources,
            cleanup_grace_s=130,
            observe_cleanup=lambda: observe_descendant_cleanup(PREPARED_ROOT),
            check_result=lambda: check_worker(registration, registration_ref),
        )
        result["owned_wrapper_ref"] = owned
        result["worker_ref"] = check_worker(registration, registration_ref)
        check_sources(sources)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        if (directory / "D1/result.json").is_file():
            result["failed_owned_wrapper_ref"] = reference(directory / "D1/result.json")
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
    print(
        json.dumps(execute(dict(path=str(args.registration.resolve()), sha256=args.registration_sha256), mode=mode)),
        flush=True,
    )


if __name__ == "__main__":
    main()
