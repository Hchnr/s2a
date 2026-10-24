"""Own038's one99-request STATE process and independent CPU state audit."""

import argparse
import fcntl
import json
import os
import signal
import socket
from pathlib import Path

from run_wam_clean_bridge import check_owned_command, resource_observation
from run_wam_clean_training import MASTER_PORT, _flags, environment, observe_release, prior
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU, admit_execution, publish, read, reference, require


def planned_worker(prepared, prepared_ref, registration_ref):
    w2 = prior(prepared, prepared_ref, registration_ref, "W2")
    command = [
        prepared["model_config"]["model_python"],
        "scripts/sample_wam_clean_training_state.py",
        *_flags("prepared", prepared_ref),
        *_flags("execution-registration", registration_ref),
        *_flags("w2-fit", w2["producer_ref"]),
        *_flags("w2-verification", w2["verification_ref"]),
        "--gpu-uuid",
        GPU,
        "--seed",
        "42",
    ]
    return command, w2


def planned_audit(prepared, prepared_ref, registration_ref, producer_ref, owned, gpu_after_ref):
    return [
        str(REPO / ".venv/bin/python"),
        "scripts/verify_wam_clean_training_state.py",
        *_flags("prepared", prepared_ref),
        *_flags("execution-registration", registration_ref),
        *_flags("sampling", producer_ref),
        *_flags("worker-launch", owned["launch"]),
        *_flags("worker-exit", owned["exit"]),
        *_flags("worker-release", owned["release"]),
        *_flags("gpu-after", gpu_after_ref),
        "--output",
        str(Path(prepared["root"]) / "review/STATE_verified.json"),
        "--seed",
        "42",
    ]


def check_artifacts(prepared_ref, registration_ref, producer_ref, w2, verification_ref=None, gpu_after_ref=None):
    produced = read(producer_ref)
    common = dict(complete=True, passed=True, prepared_ref=prepared_ref, execution_registration_ref=registration_ref)
    require(
        all(produced.get(k) == v for k, v in common.items())
        and produced.get("schema_version") == "s2a.clean-training-state-sampling.v1"
        and produced.get("actual_requests") == 99
        and produced.get("actual_denoise_calls") == 3465
        and produced.get("model_initializations") == produced.get("model_initialization_attempts") == 1
        and produced.get("w2_fit_ref") == w2["producer_ref"]
        and produced.get("w2_verification_ref") == w2["verification_ref"],
        "STATE producer identity/budget differs",
    )
    fit = read(w2["producer_ref"])
    require(produced.get("readout_ref") == fit["params_ref"], "STATE is not the single frozen W2 candidate")
    if verification_ref is not None:
        audit = read(verification_ref)
        require(
            all(audit.get(k) == v for k, v in common.items())
            and audit.get("schema_version") == "s2a.clean-training-state-verification.v1"
            and audit.get("sampling_ref") == producer_ref
            and audit.get("w2_fit_ref") == w2["producer_ref"]
            and audit.get("w2_verification_ref") == w2["verification_ref"]
            and audit.get("readout_ref") == produced["readout_ref"]
            and audit.get("gpu_after_ref") == gpu_after_ref
            and audit.get("actual_requests") == 99
            and audit.get("actual_denoise_calls") == 3465
            and audit.get("numerical_contract_passed") is True
            and audit.get("eligible_for_development") is True
            and type(audit.get("state_gate_passed")) is bool,
            "STATE full numerical audit failed; diagnostic pointwise gate cannot substitute for it",
        )


def execute(prepared_ref, registration_ref, *, dry_run=False):
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("LD_LIBRARY_PATH") == "",
        "STATE owner needs S2A cwd and empty CUDA/LD",
    )
    prepared, registration, sources = admit_execution(prepared_ref, registration_ref, required_sources=[__file__])
    require(
        registration["master_port"] == MASTER_PORT
        and registration["stage_entrypoints"]["STATE"] == "scripts/sample_wam_clean_training_state.py",
        "registered state entrypoint/port differs",
    )
    worker, w2 = planned_worker(prepared, prepared_ref, registration_ref)
    root = Path(prepared["root"])
    if dry_run:
        return dict(
            worker=worker,
            timeout_s=prepared["resources"]["state_timeout_s"],
            gpu_uuid=GPU,
            new_model_initializations=0,
            new_denoise_calls=0,
            new_fits=0,
            physics_steps=0,
            fresh_cases=0,
        )
    directory = root / "orchestration/STATE"
    require(
        not (root / "state_sampling").exists() and not (root / "review/STATE_verified.json").exists(),
        "refuses existing state generation/audit or retry",
    )
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version="s2a.clean-training-state-owner.v1",
        stage="STATE",
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        owned_commands=[],
        planned_worker=worker,
        automatic_retry=False,
        fresh_performance_authorized=False,
    )
    try:
        lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks" / f"{GPU}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = resource_observation(GPU)
            result["gpu_before_ref"] = publish(directory / "gpu_before.json", before)
            require(
                before["memory_mib"] == 0 and before["compute_apps"] == [] and before["utilization_gpu"] == 0,
                "STATE GPU occupied; no existing process is changed",
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", MASTER_PORT))
            try:
                entry = owned_command(
                    worker,
                    environment(prepared, gpu=True),
                    directory / "model",
                    timeout=prepared["resources"]["state_timeout_s"],
                )
                result["owned_commands"].append(entry)
                check_owned_command(entry)
            finally:
                if (directory / "model/ownership/release.json").exists():
                    result["gpu_after_ref"] = observe_release(directory)
        result["producer_ref"] = reference(root / "state_sampling/result.json")
        check_artifacts(prepared_ref, registration_ref, result["producer_ref"], w2)
        audit = planned_audit(
            prepared, prepared_ref, registration_ref, result["producer_ref"], entry, result["gpu_after_ref"]
        )
        entry = owned_command(
            audit,
            environment(prepared, gpu=False),
            directory / "independent",
            timeout=prepared["resources"]["audit_timeout_s"],
        )
        result["owned_commands"].append(entry)
        check_owned_command(entry)
        result["verification_ref"] = reference(root / "review/STATE_verified.json")
        check_artifacts(
            prepared_ref,
            registration_ref,
            result["producer_ref"],
            w2,
            result["verification_ref"],
            result["gpu_after_ref"],
        )
        for path, digest in sources.items():
            require(reference(path)["sha256"] == digest, "STATE source changed during execution")
        result.update(
            complete=True,
            passed=True,
            state_gate_passed=read(result["verification_ref"])["state_gate_passed"],
            numerical_contract_passed=True,
            eligible_for_development=True,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if result["passed"] is not True:
            result["available_failure_receipts"] = {
                str(path.relative_to(directory)): reference(path)
                for folder in ("model", "independent")
                for leaf in ("launch.json", "exit.json", "ownership/release.json")
                if (path := directory / folder / leaf).is_file()
            }
        publish(directory / "result.json", result)
    return reference(directory / "result.json")


def interrupted(signum, frame):
    raise InterruptedError(f"STATE owner interrupted by signal{signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution-registration"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    print(
        json.dumps(
            execute(
                dict(path=str(args.prepared.resolve()), sha256=args.prepared_sha256),
                dict(path=str(args.execution_registration.resolve()), sha256=args.execution_registration_sha256),
                dry_run=args.dry_run,
            )
        )
    )


if __name__ == "__main__":
    main()
