"""Own the four serial038 training stages, with no retry or stage skipping.

The separate final execution registration is required even for D0. Metadata
preparation alone cannot invoke this owner. State/development have their own
entrypoints; this owner never opens the reserved fresh performance collection.
"""

import argparse
import fcntl
import json
import os
import signal
import socket
import time
from pathlib import Path

from run_wam_clean_bridge import check_owned_command, resource_observation
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU, admit_execution, publish, read, reference, require

STAGES = ("D0", "W1", "D1", "W2")
MASTER_PORT = 31338


def environment(prepared, *, gpu):
    cache = Path(prepared["root"]) / "model_cache"
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=GPU if gpu else "",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=prepared["model_config"]["cosmos_source_root"]
        + ":"
        + str(REPO / "src")
        + ":"
        + str(REPO / "scripts"),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(MASTER_PORT),
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
        LOCAL_WORLD_SIZE="1",
        COSMOS_TRAINING="0",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        XDG_CACHE_HOME=str(cache),
        HF_HOME=str(cache / "hf"),
        TORCH_HOME=str(cache / "torch"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        NUMBA_CACHE_DIR=str(cache / "numba"),
        MPLCONFIGDIR=str(cache / "mpl"),
    )


def _flags(name, ref):
    return [f"--{name}", ref["path"], f"--{name}-sha256", ref["sha256"]]


def check_stage_artifacts(stage, prepared_ref, registration_ref, producer_ref, verification_ref=None):
    """Bind the exact stage and producer, rather than accepting generic True."""
    producer = read(producer_ref)
    common = dict(prepared_ref=prepared_ref, execution_registration_ref=registration_ref, complete=True, passed=True)
    require(all(producer.get(k) == v for k, v in common.items()), "producer failed or belongs to another run")
    capture = stage in ("D0", "D1")
    require(
        producer.get("schema_version") == ("s2a.clean-training-capture.v1" if capture else "s2a.clean-training-fit.v1")
        and producer.get("round_name" if capture else "stage") == stage,
        "producer schema/stage differs",
    )
    if capture:
        require(
            producer.get("actual_requests") == 300
            and producer.get("actual_denoise_calls") == 10500
            and producer.get("future_label_reads") == 0,
            "producer capture budget differs",
        )
    else:
        require(
            producer.get("producer_fits") == 1
            and producer.get("actual_rows") == (420000 if stage == "W1" else 840000)
            and producer.get("rounds") == (["D0"] if stage == "W1" else ["D0", "D1"]),
            "producer fit budget differs",
        )
    if verification_ref is not None:
        audit = read(verification_ref)
        require(
            all(audit.get(k) == v for k, v in common.items()),
            "independent stage audit failed or belongs to another run",
        )
        require(
            audit.get("schema_version")
            == ("s2a.clean-training-capture-audit.v1" if capture else "s2a.clean-training-fit-verification.v1")
            and audit.get("round_name" if capture else "stage") == stage
            and audit.get("collection_ref" if capture else "fit_ref") == producer_ref,
            "independent audit does not bind this exact stage/producer",
        )
        if not capture:
            require(audit.get("params_ref") == producer.get("params_ref"), "independent fit parameters differ")


def prior(prepared, prepared_ref, registration_ref, stage):
    value = read(reference(Path(prepared["root"]) / "orchestration" / stage / "result.json"))
    require(
        value.get("complete") is True
        and value.get("passed") is True
        and value.get("stage") == stage
        and value.get("prepared_ref") == prepared_ref
        and value.get("execution_registration_ref") == registration_ref,
        "prior stage owner is incomplete or unrelated",
    )
    for command in value["owned_commands"]:
        check_owned_command(command)
    require(len(value["owned_commands"]) == 2, "prior stage lacks production and independent owned commands")
    check_stage_artifacts(stage, prepared_ref, registration_ref, value["producer_ref"], value["verification_ref"])
    return value


def planned_worker(prepared, prepared_ref, registration_ref, stage):
    require(stage in STAGES, "unregistered training stage")
    root = Path(prepared["root"])
    common = _flags("prepared", prepared_ref) + _flags("execution-registration", registration_ref) + ["--seed", "42"]
    python = prepared["model_config"]["model_python"] if stage in ("D0", "D1") else str(REPO / ".venv/bin/python")
    if stage in ("D0", "D1"):
        command = [python, "scripts/collect_wam_clean_training.py", *common, "--round", stage, "--gpu-uuid", GPU]
        if stage == "D1":
            d0 = prior(prepared, prepared_ref, registration_ref, "D0")
            w1 = prior(prepared, prepared_ref, registration_ref, "W1")
            command += _flags("d0-audit", d0["verification_ref"]) + _flags("w1-fit", w1["producer_ref"])
            command += _flags("w1-verification", w1["verification_ref"])
        return command, root / "capture" / stage / "result.json"
    rounds = ("D0",) if stage == "W1" else ("D0", "D1")
    previous = [prior(prepared, prepared_ref, registration_ref, name) for name in rounds]
    command = [
        python,
        "scripts/fit_wam_clean_training.py",
        *common,
        "--stage",
        stage,
        "--output-dir",
        str(root / "fit" / stage),
    ]
    for value in previous:
        command += ["--collection", value["producer_ref"]["path"], value["producer_ref"]["sha256"]]
        command += ["--capture-audit", value["verification_ref"]["path"], value["verification_ref"]["sha256"]]
    return command, root / "fit" / stage / "result.json"


def planned_audit(
    prepared, prepared_ref, registration_ref, stage, producer_ref, model_command=None, gpu_after_ref=None
):
    common = _flags("prepared", prepared_ref) + _flags("execution-registration", registration_ref) + ["--seed", "42"]
    root = Path(prepared["root"])
    output = root / "review" / f"{stage}_verified.json"
    python = str(REPO / ".venv/bin/python")
    if stage in ("W1", "W2"):
        return [
            python,
            "scripts/verify_wam_clean_training_fit.py",
            *common,
            "--stage",
            stage,
            *_flags("fit", producer_ref),
            "--output",
            str(output),
        ], output
    require(model_command is not None and gpu_after_ref is not None, "capture audit requires real worker ownership")
    command = [
        python,
        "scripts/verify_wam_clean_training_capture.py",
        *common,
        *_flags("collection", producer_ref),
        *_flags("worker-launch", model_command["launch"]),
        *_flags("worker-exit", model_command["exit"]),
        *_flags("worker-release", model_command["release"]),
        *_flags("gpu-after", gpu_after_ref),
        "--output",
        str(output),
    ]
    if stage == "D1":
        command += _flags("d0-audit", prior(prepared, prepared_ref, registration_ref, "D0")["verification_ref"])
    return command, output


def observe_release(directory, *, timeout=120):
    require(timeout == 120, "registered release budget is120 seconds")
    start = time.monotonic()
    with (directory / "resource_observations.jsonl").open("x") as stream:
        while True:
            value = resource_observation(GPU)
            value["seconds_since_actual_owned_wait"] = time.monotonic() - start
            stream.write(json.dumps(value) + "\n")
            stream.flush()
            if value["memory_mib"] == 0 and value["compute_apps"] == []:
                return publish(directory / "gpu_after.json", dict(value, released=True))
            if time.monotonic() - start >= timeout:
                raise TimeoutError("038 owned worker exited; GPU release was not observed within120 seconds")
            time.sleep(0.25)


def execute(prepared_ref, registration_ref, *, stage, dry_run=False):
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("LD_LIBRARY_PATH") == "",
        "038 owner needs S2A cwd and explicit empty CUDA/LD",
    )
    prepared, registration, sources = admit_execution(prepared_ref, registration_ref, required_sources=[__file__])
    require(registration["master_port"] == MASTER_PORT, "registered private port differs")
    worker, result_path = planned_worker(prepared, prepared_ref, registration_ref, stage)
    require(registration["stage_entrypoints"][stage] == worker[1], "registered stage entrypoint differs")
    if dry_run:
        return dict(
            stage=stage,
            worker=worker,
            timeout_s=prepared["resources"][dict(D0="d0_timeout_s", D1="d1_timeout_s").get(stage, "fit_timeout_s")],
            cuda=GPU if stage in ("D0", "D1") else "",
            new_model_calls=0,
            new_fits=0,
        )
    directory = Path(prepared["root"]) / "orchestration" / stage
    require(
        not result_path.exists() and not (Path(prepared["root"]) / "review" / f"{stage}_verified.json").exists(),
        "refuses existing stage artifacts or retry",
    )
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version="s2a.clean-training-owner.v1",
        stage=stage,
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
        if stage in ("D0", "D1"):
            lock_path = REPO / "outputs/wam_stability_20261021/gpu_locks" / f"{GPU}.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                before = resource_observation(GPU)
                result["gpu_before_ref"] = publish(directory / "gpu_before.json", before)
                require(
                    before["memory_mib"] == 0 and not before["compute_apps"] and before["utilization_gpu"] == 0,
                    "registered GPU occupied; existing processes remain untouched",
                )
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", MASTER_PORT))
                try:
                    entry = owned_command(
                        worker,
                        environment(prepared, gpu=True),
                        directory / "model",
                        timeout=prepared["resources"][stage.lower() + "_timeout_s"],
                    )
                    result["owned_commands"].append(entry)
                    check_owned_command(entry)
                finally:
                    if (directory / "model/ownership/release.json").exists():
                        result["gpu_after_ref"] = observe_release(directory)
        else:
            entry = owned_command(
                worker,
                environment(prepared, gpu=False),
                directory / "producer",
                timeout=prepared["resources"]["fit_timeout_s"],
            )
            result["owned_commands"].append(entry)
            check_owned_command(entry)
        result["producer_ref"] = reference(result_path)
        check_stage_artifacts(stage, prepared_ref, registration_ref, result["producer_ref"])
        audit, audit_path = planned_audit(
            prepared,
            prepared_ref,
            registration_ref,
            stage,
            result["producer_ref"],
            model_command=entry,
            gpu_after_ref=result.get("gpu_after_ref"),
        )
        timeout = prepared["resources"]["audit_timeout_s" if stage in ("D0", "D1") else "independent_fit_timeout_s"]
        entry = owned_command(audit, environment(prepared, gpu=False), directory / "independent", timeout=timeout)
        result["owned_commands"].append(entry)
        check_owned_command(entry)
        result["verification_ref"] = reference(audit_path)
        check_stage_artifacts(stage, prepared_ref, registration_ref, result["producer_ref"], result["verification_ref"])
        for path, digest in sources.items():
            require(reference(path)["sha256"] == digest, "source changed during owned stage")
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if result["passed"] is not True:
            result["available_failure_receipts"] = {
                str(path.relative_to(directory)): reference(path)
                for folder in ("model", "producer", "independent")
                for leaf in ("launch.json", "exit.json", "ownership/release.json")
                if (path := directory / folder / leaf).is_file()
            }
        publish(directory / "result.json", result)
    return reference(directory / "result.json")


def interrupted(signum, frame):
    raise InterruptedError(f"038 owner interrupted by signal{signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution-registration"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
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
                stage=args.stage,
                dry_run=args.dry_run,
            )
        )
    )


if __name__ == "__main__":
    main()
