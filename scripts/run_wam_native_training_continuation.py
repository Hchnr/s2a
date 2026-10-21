"""Own one prefix audit, native tasks1/2 and one final continuation audit."""

import argparse
import json
import os
import signal
from pathlib import Path

from prepare_wam_native_training_continuation import read_prepared
from run_wam_native_training import attempt, preserve_attempt_evidence, read, reference, require

from s2a.artifacts import REPO
from s2a.worker_handshake import write_once

AUDIT_SCHEMA = "s2a.native-training-continuation-independent-verification.v1"


def admission(args):
    require(
        Path.cwd().resolve() == REPO
        and args.seed == 42
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == ""
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "continuation owner requires S2A cwd, seed42 and empty CUDA/LD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    require(
        prepared["schema_version"] == "s2a.native-training-continuation-preparation.v1"
        and prepared["seed"] == 42
        and prepared["remaining_tasks"] == [1, 2]
        and len(prepared["records"]) == 30
        and len(prepared["cached_records"]) == 10
        and len(prepared["remaining_records"]) == 20
        and prepared["cached_records"] == [r for r in prepared["records"] if r["task_index"] == 0]
        and prepared["remaining_records"] == [r for r in prepared["records"] if r["task_index"] in (1, 2)]
        and all(sum(r["task_index"] == i for r in prepared["records"]) == 10 for i in range(3)),
        "continuation must preserve cached10 and the original remaining20 roster",
    )
    totals = [sum(r["native_steps"] + 40 for r in prepared["records"] if r["task_index"] == i) for i in range(3)]
    require(totals == [1328, 2214, 1438], "continuation original fixed physics budget changed")
    require(
        prepared["original_task_passed"] is prepared["original_owner_passed"] is False
        and prepared["cached_quality_verified"] is False
        and prepared["model_training_authorized"] is prepared["fresh_performance_authorized"] is False,
        "continuation cannot recertify the failed original owner or authorize model training",
    )
    root = Path(prepared["output_root"])
    require(root.is_relative_to(REPO / "outputs"), "continuation output must remain under S2A outputs")
    return prepared


def planned_commands(args, prepared):
    root = Path(prepared["output_root"])
    common = ["--prepared", str(args.prepared.resolve()), "--prepared-sha256", args.prepared_sha256]
    checker = [str(REPO / ".venv/bin/python"), "scripts/verify_wam_native_training_continuation.py"]
    prefix = [
        *checker,
        "--phase",
        "prefix",
        *common,
        "--output",
        str(root / "review/prefix_verified.json"),
        "--seed",
        "42",
    ]
    # Preserve the venv entry point: resolving its symlink loses the simulation environment.
    simulator = str(REPO / prepared["original_config"]["simulation_python"])
    workers = [
        [
            simulator,
            "scripts/collect_wam_native_training_continuation.py",
            *common,
            "--task-index",
            str(i),
            "--seed",
            "42",
        ]
        for i in (1, 2)
    ]
    final = [
        *checker,
        "--phase",
        "collection",
        *common,
        "--prefix-verification",
        str(root / "review/prefix_verified.json"),
        "--collection",
        str(root / "collection/result.json"),
        "--output",
        str(root / "review/collection_verified.json"),
        "--seed",
        "42",
    ]
    return prefix, workers, final


def completed_audit(path, args, phase):
    audit = read(path)
    require(
        audit.get("schema_version") == AUDIT_SCHEMA
        and audit.get("phase") == phase
        and audit.get("passed") is audit.get(f"{phase}_verified") is True
        and audit.get("prepared") == reference(args.prepared)
        and audit.get("input_sha256", {}).get(str(args.prepared.resolve())) == args.prepared_sha256
        and audit.get("model_training_authorized") is audit.get("fresh_performance_authorized") is False,
        f"continuation {phase} audit missing, incomplete or unrelated",
    )
    return audit


def completed_task(path, task_index, args, prepared):
    task = read(path)
    expected_steps = sum(r["native_steps"] + 40 for r in prepared["remaining_records"] if r["task_index"] == task_index)
    require(
        task.get("schema_version") == "s2a.native-training-continuation-task.v1"
        and task.get("complete") is task.get("passed") is True
        and "error" not in task
        and task.get("task_index") == task_index
        and task.get("prepared_ref") == prepared["original_prepared_ref"]
        and task.get("continuation_prepared_ref") == reference(args.prepared)
        and len(task.get("cases", [])) == task.get("environment_initializations") == 10
        and task.get("current_canvas_images") == 50
        and task.get("physics_steps") == expected_steps
        and task.get("cached_cases_recollected") == 0,
        "continuation task incomplete or original source/budget changed",
    )
    require(
        all(
            task.get(k) == 0
            for k in (
                "wam_initializations",
                "denoise_calls",
                "fits",
                "fresh_cases",
                "future_rgb_frames",
                "source_hdf5_reads",
            )
        ),
        "continuation task exceeded zero model/fresh/HDF budget",
    )
    readiness_ref = task["environment_readiness_ref"]
    readiness = read(readiness_ref["path"], readiness_ref["sha256"])
    require(
        readiness.get("schema_version") == "s2a.native-training-environment-readiness.v1"
        and readiness.get("complete") is readiness.get("passed") is True
        and readiness.get("continuation_prepared_ref") == reference(args.prepared)
        and readiness.get("restore_performed") is True
        and readiness.get("after_restore") == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(REPO)),
        "continuation task lacks explicit successful local environment restoration",
    )
    return task


def execute(args):
    prepared = admission(args)
    prefix_command, workers, final_command = planned_commands(args, prepared)
    root = Path(prepared["output_root"])
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    prefix=prefix_command,
                    workers=workers,
                    verifier=final_command,
                    new_cases=20,
                    cached_cases=10,
                    model_calls=0,
                    cuda_visible_devices="",
                )
            )
        )
        return
    directory, collection, review = root / "orchestration/collection", root / "collection", root / "review"
    require(
        not any(p.exists() for p in (directory, collection, review)),
        "continuation refuses existing collection/owner/review output",
    )
    directory.mkdir(parents=True)
    collection.mkdir()
    review.mkdir()
    result = dict(
        schema_version="s2a.native-training-continuation-owner.v1",
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        original_prepared_ref=prepared["original_prepared_ref"],
        cached_task_ref=prepared["failed_task_ref"],
        original_task_passed=False,
        original_owner_passed=False,
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
        attempt(result, prefix_command, directory / "prefix", "prefix_audit")
        prefix_path = review / "prefix_verified.json"
        prefix = completed_audit(prefix_path, args, "prefix")
        require(
            len(prefix.get("case_verification", [])) == len(prefix.get("cached_case_refs", [])) == 10
            and prefix.get("original_task_passed") is prefix.get("original_owner_passed") is False
            and prefix.get("original_worker_returncode") == 1
            and prefix.get("eligible_for_model_training_registration") is False,
            "continuation prefix must independently verify ten saved cases and retain original failure",
        )
        result["prefix_verification_ref"] = reference(prefix_path)
        tasks, task_results = [], []
        for index, command in zip((1, 2), workers, strict=True):
            admission(args)
            lifecycle = attempt(
                result,
                command,
                directory / f"task{index}",
                "collection",
                index,
                started=collection / f"task{index}/worker_started.json",
            )
            path = collection / f"task{index}/result.json"
            tasks.append(completed_task(path, index, args, prepared))
            task_results.append(dict(task_index=index, result=reference(path), lifecycle=lifecycle))
        new_steps = sum(task["physics_steps"] for task in tasks)
        require(new_steps == 3652, "continuation remaining20 physics budget exceeded")
        public = dict(
            schema_version="s2a.native-training-continuation-collection.v1",
            complete=True,
            passed=True,
            prepared_ref=reference(args.prepared),
            original_prepared_ref=prepared["original_prepared_ref"],
            cached_task_ref=prepared["failed_task_ref"],
            prefix_verification_ref=reference(prefix_path),
            task_results=task_results,
            cached_cases=10,
            cached_physics_steps=1328,
            cached_current_canvas_images=50,
            physics_steps=4980,
            environment_initializations=30,
            current_canvas_images=150,
            new_collection_physics_steps=new_steps,
            new_environment_initializations=20,
            new_current_canvas_images=100,
            wam_initializations=0,
            denoise_calls=0,
            fits=0,
            fresh_cases=0,
            automatic_retry=False,
            model_training_authorized=False,
            fresh_performance_authorized=False,
        )
        path = collection / "result.json"
        write_once(path, public)
        result["producer_result"] = reference(path)
        attempt(result, final_command, directory / "independent", "independent_audit")
        audit_path = review / "collection_verified.json"
        audit = completed_audit(audit_path, args, "collection")
        require(
            audit.get("input_sha256", {}).get(str(path.resolve())) == result["producer_result"]["sha256"]
            and audit.get("prefix_verification_ref") == reference(prefix_path)
            and len(audit.get("case_verification", [])) == 30
            and audit.get("physics_steps") == 4980
            and audit.get("new_collection_physics_steps") == 3652
            and audit.get("environment_initializations") == 30
            and audit.get("current_canvas_images") == 150
            and isinstance(audit.get("eligible_for_model_training_registration"), bool),
            "continuation final audit lost full denominator or source/lifecycle binding",
        )
        admission(args)
        result.update(
            complete=True,
            passed=True,
            verification_ref=reference(audit_path),
            physics_steps=4980,
            new_collection_physics_steps=3652,
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
    raise InterruptedError(f"continuation CPU owner interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    execute(args)


if __name__ == "__main__":
    main()
