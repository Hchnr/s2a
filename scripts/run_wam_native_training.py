"""Own the registered030 CPU-only synchronous native collection and audit."""

import argparse
import json
import os
import signal
from pathlib import Path

from prepare_wam_native_training import read_prepared
from run_wam_readout import owned_command

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def read(path, expected=None):
    require(expected is None or sha256(path) == expected, "030 evidence SHA changed")
    return json.loads(Path(path).read_text())


def cpu_environment():
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )


def admission(args):
    require(
        Path.cwd().resolve() == REPO
        and args.seed == 42
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "030 owner requires repository cwd and empty CUDA/LD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    peer = read(args.source_peer, args.source_peer_sha256)
    require(
        peer.get("schema_version") == "s2a.native-training-independent-verification.v1"
        and peer.get("phase") == "prepare"
        and peer.get("passed") is peer.get("preparation_verified") is True
        and peer.get("prepared") == reference(args.prepared)
        and peer.get("input_sha256", {}).get(str(args.prepared.resolve())) == args.prepared_sha256
        and peer.get("model_training_authorized") is peer.get("fresh_performance_authorized") is False,
        "030 complete independent metadata admission required",
    )
    for path, digest in peer["input_sha256"].items():
        require(sha256(path) == digest, "030 independent source admission input changed")
    require(
        len(prepared["records"]) == 30
        and all(sum(r["task_index"] == i for r in prepared["records"]) == 10 for i in range(3)),
        "030 fixed three-task/30-case roster required",
    )
    root = Path(prepared["output_root"])
    require(root.is_relative_to(REPO / "outputs"), "030 output must remain under S2A outputs")
    return prepared, peer


def planned_commands(args, prepared):
    root = Path(prepared["output_root"])
    # Keep the configured venv entry point rather than resolving its symlink.
    simulator = str(REPO / prepared["config"]["simulation_python"])
    common = ["--prepared", str(args.prepared.resolve()), "--prepared-sha256", args.prepared_sha256]
    workers = [
        [simulator, "scripts/collect_wam_native_training.py", *common, "--task-index", str(i), "--seed", "42"]
        for i in range(3)
    ]
    verifier = [
        str(REPO / ".venv/bin/python"),
        "scripts/verify_wam_native_training.py",
        "--phase",
        "collection",
        *common,
        "--collection",
        str(root / "collection/result.json"),
        "--output",
        str(root / "review/collection_verified.json"),
        "--seed",
        "42",
    ]
    return workers, verifier


def completed_task(path, task_index, prepared_path):
    result = read(path)
    require(
        result.get("schema_version") == "s2a.native-training-task.v1"
        and result.get("complete") is result.get("passed") is True
        and "error" not in result
        and result["task_index"] == task_index
        and result["prepared_ref"] == reference(prepared_path)
        and len(result["cases"]) == result["environment_initializations"] == 10
        and result["current_canvas_images"] == 50
        and 830 <= result["physics_steps"] <= 6400,
        "030 task producer incomplete or outside fixed case/image/physics budget",
    )
    require(
        all(result[k] == 0 for k in ("wam_initializations", "denoise_calls", "fits", "fresh_cases")),
        "030 CPU data collection exceeded zero-model/fresh budget",
    )
    return result


def attempt(result, command, directory, phase, task_index=None, started=None):
    item = dict(phase=phase, command=command, directory=str(directory), task_index=task_index)
    result["command_attempts"].append(item)
    completed = owned_command(command, cpu_environment(), directory, started=started)
    result["owned_commands"].append(completed)
    item["completed_lifecycle"] = completed
    return completed


def preserve_attempt_evidence(result):
    """Retain a failed actual child even when owned_command raises before return."""
    for item in result["command_attempts"]:
        directory = Path(item["directory"])
        item["observed_lifecycle"] = {
            name: reference(directory / suffix)
            for name, suffix in (
                ("launch", "launch.json"),
                ("exit", "exit.json"),
                ("release", "ownership/release.json"),
            )
            if (directory / suffix).exists()
        }


def execute(args):
    prepared, _ = admission(args)
    workers, verifier = planned_commands(args, prepared)
    root = Path(prepared["output_root"])
    if args.dry_run:
        print(json.dumps(dict(workers=workers, verifier=verifier, model_calls=0, cuda_visible_devices="")))
        return
    directory, collection = root / "orchestration/collection", root / "collection"
    require(not directory.exists() and not collection.exists(), "030 refuses existing collection/owner output")
    directory.mkdir(parents=True)
    collection.mkdir()
    result = dict(
        schema_version="s2a.native-training-owner.v1",
        stage="collection",
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        source_peer_ref=reference(args.source_peer),
        command_attempts=[],
        owned_commands=[],
        automatic_retry=False,
        eligible_for_model_training_registration=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        wam_initializations=0,
        denoise_calls=0,
        fits=0,
        fresh_cases=0,
    )
    try:
        tasks, task_results = [], []
        for index, command in enumerate(workers):
            read_prepared(args.prepared, args.prepared_sha256)
            lifecycle = attempt(
                result,
                command,
                directory / f"task{index}",
                "collection",
                index,
                started=collection / f"task{index}/worker_started.json",
            )
            path = collection / f"task{index}/result.json"
            tasks.append(completed_task(path, index, args.prepared))
            task_results.append(dict(task_index=index, result=reference(path), lifecycle=lifecycle))
        public = dict(
            schema_version="s2a.native-training-collection.v1",
            complete=True,
            passed=True,
            prepared_ref=reference(args.prepared),
            task_results=task_results,
            physics_steps=sum(t["physics_steps"] for t in tasks),
            environment_initializations=30,
            current_canvas_images=150,
            wam_initializations=0,
            denoise_calls=0,
            fits=0,
            fresh_cases=0,
            automatic_retry=False,
            model_training_authorized=False,
            fresh_performance_authorized=False,
        )
        require(public["physics_steps"] <= 19200, "030 aggregate physics budget exceeded")
        path = collection / "result.json"
        write_once(path, public)
        result["producer_result"] = reference(path)
        attempt(result, verifier, directory / "independent", "independent_audit")
        path = root / "review/collection_verified.json"
        audit = read(path)
        require(
            audit.get("schema_version") == "s2a.native-training-independent-verification.v1"
            and audit.get("phase") == "collection"
            and audit.get("passed") is audit.get("collection_verified") is True
            and audit.get("prepared") == reference(args.prepared)
            and audit["input_sha256"].get(str((collection / "result.json").resolve()))
            == result["producer_result"]["sha256"]
            and audit.get("model_training_authorized") is audit.get("fresh_performance_authorized") is False,
            "030 complete independent collection audit missing or unrelated",
        )
        admission(args)
        result.update(
            complete=True,
            passed=True,
            verification_ref=reference(path),
            physics_steps=public["physics_steps"],
            eligible_for_model_training_registration=audit["eligible_for_model_training_registration"],
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        preserve_attempt_evidence(result)
        write_once(directory / "result.json", result)
    print(json.dumps(dict(result=reference(directory / "result.json"))))


def interrupted(signum, frame):
    raise InterruptedError(f"030 CPU owner interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--source-peer", type=Path, required=True)
    parser.add_argument("--source-peer-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    execute(args)


if __name__ == "__main__":
    main()
