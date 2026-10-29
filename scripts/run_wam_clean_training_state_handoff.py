"""048 one-shot durable CPU owner of the original STATE model/audit chain."""

import argparse
import json
import os
import signal
from pathlib import Path

from s2a.infrastructure.lifecycle.durable_stage_supervisor import (
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
from s2a.research.wam.training.clean.clean_training_state_handoff import PREFIX, ROOT, absent_outputs, admit
from s2a.research.wam.training.clean.clean_training_w1_handoff import PREPARED_ROOT
from s2a.research.wam.training.state_worker_checks import check_worker, worker_argv
from scripts.run_wam_clean_training_continuation import observe_descendant_cleanup

ENTRY = "scripts/run_wam_clean_training_state_handoff.py"
WORKER = "scripts/execute_wam_clean_training_state_handoff.py"


def argv(ref, *mode):
    return [
        str(REPO / ".venv/bin/python"),
        ENTRY,
        "--registration",
        ref["path"],
        "--registration-sha256",
        ref["sha256"],
        "--seed",
        "42",
        *mode,
    ]


def interrupted(signum, frame):
    raise InterruptedError(f"048 supervisor interrupted by signal {signum}")


def execute(ref, *, mode):
    check_cpu_environment(os.environ)
    require(Path.cwd().resolve() == REPO, "048 supervisor must run from S2A")
    a = admit(ref, role="owner")
    sources = a["source_sha256"]
    for relative in (ENTRY, WORKER):
        require(
            sources.get(str(REPO / relative)) == reference(REPO / relative)["sha256"],
            "048 actual owner/worker source is not frozen",
        )
    if mode == "dry-run":
        return dict(
            schema_version=PREFIX + "owner-dry.v1",
            complete=True,
            passed=True,
            registration_ref=ref,
            command=worker_argv(ref),
            timeout_s=a["owner_timeout_s"],
            cleanup_grace_s=130,
            actual_model_calls=0,
            actual_worker_launches=0,
        )
    directory = ROOT / "supervisor"
    if mode == "launch":
        absent_outputs()
        return launch_detached(
            argv(ref, "--run"), directory=directory, source_sha256=sources, environment=cpu_environment()
        )
    require(mode == "run", "048 supervisor mode differs")
    launch = accept_detached_launch(directory, command=argv(ref, "--run"), source_sha256=sources)
    result = dict(
        schema_version=PREFIX + "owner.v1",
        complete=False,
        passed=False,
        registration_ref=ref,
        identity=identity(os.getpid()),
        launch_ref=launch,
        stage="STATE",
        w2_owner_ref=a["w2_owner_ref"],
        w2_params_ref=a["w2_params_ref"],
        automatic_retry=False,
        development_authorized=False,
        fresh_performance_authorized=False,
    )
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        owned = owned_stage(
            worker_argv(ref),
            environment=cpu_environment(),
            directory=directory / "STATE",
            timeout_s=a["owner_timeout_s"],
            source_sha256=sources,
            cleanup_grace_s=130,
            observe_cleanup=lambda: observe_descendant_cleanup(PREPARED_ROOT),
            check_result=lambda: check_worker(a, ref),
        )
        worker_ref = check_worker(a, ref)
        worker = read(worker_ref)
        check_sources(sources)
        result.update(
            owned_wrapper_ref=owned,
            worker_ref=worker_ref,
            state_owner_ref=worker["delegate_result_ref"],
            verification_ref=worker["verification_ref"],
            state_gate_passed=worker["state_gate_passed"],
            numerical_contract_passed=True,
            eligible_for_development=True,
            complete=True,
            passed=True,
        )
    except BaseException as failure:
        result.update(complete=False, passed=False, error=dict(type=type(failure).__name__, message=str(failure)))
        if (directory / "STATE/result.json").is_file():
            result["failed_wrapper_owner_ref"] = reference(directory / "STATE/result.json")
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
