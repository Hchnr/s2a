"""Own043's one CPU W1 producer and one independent QR, without new sampling."""

import argparse
import json
import os
import signal
from pathlib import Path

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

ENTRY = "scripts/run_wam_clean_training_postcapture.py"
WORKER = "scripts/execute_wam_clean_training_postcapture_fit.py"
ROLES = ("producer", "independent")


def interrupted(signum, frame):
    raise InterruptedError(f"043 CPU owner interrupted by signal{signum}")


def argv(repair_ref, *mode):
    return [
        str(REPO / ".venv/bin/python"),
        ENTRY,
        "--repair-registration",
        repair_ref["path"],
        "--repair-registration-sha256",
        repair_ref["sha256"],
        "--seed",
        "42",
        *mode,
    ]


def worker_argv(repair_ref, role):
    require(role in ROLES, "043 has exactly two CPU roles")
    return [
        str(REPO / ".venv/bin/python"),
        WORKER,
        "--repair-registration",
        repair_ref["path"],
        "--repair-registration-sha256",
        repair_ref["sha256"],
        "--role",
        role,
        "--seed",
        "42",
    ]


def check_worker(registration, repair_ref, role):
    path = Path(registration["root"]) / "workers" / role / "result.json"
    ref = reference(path)
    value = read(ref)
    require(
        value.get("schema_version") == "s2a.clean-training-postcapture-worker.v1"
        and value.get("complete") is value.get("passed") is True
        and value.get("repair_ref") == repair_ref
        and value.get("role") == role
        and value.get("prepared_ref") == registration["prepared_ref"]
        and value.get("execution_registration_ref") == registration["execution_registration_ref"]
        and value.get("source_before_after_exact") is True
        and value.get("automatic_retry") is False,
        "043 CPU wrapper did not complete the exact admitted role",
    )
    proof = value.get("source_proof", {})
    require(
        proof.get("registration_ref") == registration["execution_registration_ref"]
        and proof.get("callback_corrected_count") == 1
        and proof.get("source_guard_completed") is True,
        "043 wrapper source callback was skipped, repeated or not restored",
    )
    require(
        value.get("producer_fits") == (1 if role == "producer" else 0)
        and value.get("independent_reconstructions") == (1 if role == "independent" else 0)
        and all(
            value.get(key) == 0
            for key in (
                "new_model_initializations",
                "new_denoise_calls",
                "new_physics_steps",
                "new_fresh_cases",
            )
        ),
        "043 wrapper exceeded its CPU-only role budget",
    )
    expected = (
        Path(registration["fit_output_dir"]) / "result.json"
        if role == "producer"
        else Path(registration["verification_output"])
    )
    delegate_ref = reference(expected)
    require(value.get("delegate_result_ref") == delegate_ref, "043 wrapper returned a different original artifact")
    delegate = read(delegate_ref)
    require(
        delegate.get("schema_version")
        == ("s2a.clean-training-fit.v1" if role == "producer" else "s2a.clean-training-fit-verification.v1")
        and delegate.get("complete") is delegate.get("passed") is True
        and delegate.get("stage") == "W1"
        and delegate.get("prepared_ref") == registration["prepared_ref"]
        and delegate.get("execution_registration_ref") == registration["execution_registration_ref"]
        and delegate.get("actual_rows") == 420000,
        "original W1 artifact is incomplete or changed its stage, inputs or denominator",
    )
    if role == "producer":
        require(
            delegate.get("producer_fits") == 1 and delegate.get("rounds") == ["D0"],
            "W1 producer is not the single fixed D0 fit",
        )
    else:
        fit_ref = reference(Path(registration["fit_output_dir"]) / "result.json")
        fit = read(fit_ref)
        require(
            delegate.get("fit_ref") == fit_ref
            and delegate.get("params_ref") == fit.get("params_ref")
            and delegate.get("independent_reconstructions") == 1
            and delegate.get("full_training_row_reconstruction") is True,
            "W1 independent QR does not bind every row of this exact producer",
        )
    return ref


def execute(repair_ref, *, mode):
    require(Path.cwd().resolve() == REPO, "043 owner must run from the current S2A repository")
    check_cpu_environment(os.environ)
    from execute_wam_clean_training_postcapture_fit import admit

    registration = admit(repair_ref)
    sources = registration["source_sha256"]
    for path in (REPO / ENTRY, REPO / WORKER):
        require(sources.get(str(path)) == reference(path)["sha256"], "actual043 owner/worker source is not frozen")
    root = Path(registration["root"])
    prepared = read(registration["prepared_ref"])
    timeouts = dict(
        producer=prepared["resources"]["fit_timeout_s"],
        independent=prepared["resources"]["independent_fit_timeout_s"],
    )
    if mode == "dry-run":
        return dict(
            schema_version="s2a.clean-training-postcapture-owner-dry.v1",
            complete=True,
            passed=True,
            repair_ref=repair_ref,
            commands={role: worker_argv(repair_ref, role) for role in ROLES},
            timeouts_s=timeouts,
            actual_worker_launches=0,
            new_model_calls=0,
            new_fits=0,
            new_physics_steps=0,
        )
    directory = root / "supervisor"
    if mode == "launch":
        require(
            not Path(registration["fit_output_dir"]).exists()
            and not Path(registration["verification_output"]).exists()
            and all(not (root / "workers" / role).exists() for role in ROLES),
            "043 actual outputs already exist; refuses retry",
        )
        return launch_detached(
            argv(repair_ref, "--run"),
            directory=directory,
            source_sha256=sources,
            environment=cpu_environment(),
        )
    require(mode == "run", "043 owner mode differs")
    launch = accept_detached_launch(directory, command=argv(repair_ref, "--run"), source_sha256=sources)
    result = dict(
        schema_version="s2a.clean-training-postcapture-owner.v1",
        complete=False,
        passed=False,
        stage="W1",
        repair_ref=repair_ref,
        prepared_ref=registration["prepared_ref"],
        execution_registration_ref=registration["execution_registration_ref"],
        retained_d0_owner_ref=registration["d0_owner_ref"],
        original_failure_ref=registration["failure_ref"],
        launch_ref=launch,
        identity=identity(os.getpid()),
        owned_commands=[],
        automatic_retry=False,
        fresh_performance_authorized=False,
        d1_authorized=False,
        source_sha256=sources,
    )
    previous_signals = {value: signal.signal(value, interrupted) for value in (signal.SIGTERM, signal.SIGINT)}
    try:
        for role in ROLES:
            result["active_role"] = role
            owner_ref = owned_stage(
                worker_argv(repair_ref, role),
                environment=cpu_environment(),
                directory=directory / role,
                timeout_s=timeouts[role],
                source_sha256=sources,
                check_result=lambda role=role: check_worker(registration, repair_ref, role),
            )
            result["owned_commands"].append(dict(role=role, owner_ref=owner_ref))
        result["producer_ref"] = reference(Path(registration["fit_output_dir"]) / "result.json")
        result["verification_ref"] = reference(Path(registration["verification_output"]))
        check_sources(sources)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        role_result = directory / result["active_role"] / "result.json"
        if role_result.is_file():
            result["failed_role_owner_ref"] = reference(role_result)
        raise
    finally:
        result["result_ref"] = str(directory / "result.json")
        try:
            publish(directory / "result.json", result)
        finally:
            for value, previous in previous_signals.items():
                signal.signal(value, previous)
    return reference(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-registration", type=Path, required=True)
    parser.add_argument("--repair-registration-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "launch", "run"):
        group.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    mode = "dry-run" if args.dry_run else "launch" if args.launch else "run"
    value = execute(
        dict(path=str(args.repair_registration.resolve()), sha256=args.repair_registration_sha256), mode=mode
    )
    print(json.dumps(value, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
