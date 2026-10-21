#!/usr/bin/env python3
"""Continue only native tasks1/2 after explicit CPU import admission."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_wam_native_training import collect_case, reference, require, write_once  # noqa: E402

from s2a.native_environment import (  # noqa: E402
    EnvironmentAdmissionError,
    environment_snapshot,
    restore_cv2_environment,
    strict_native_environment,
)


def task_records(prepared, original, task_index):
    require(task_index in (1, 2), "continuation cannot collect cached task0")
    require(
        prepared["records"] == original["records"]
        and prepared["remaining_records"]
        == [record for record in original["records"] if record["task_index"] in (1, 2)]
        and prepared["cached_records"] == [record for record in original["records"] if record["task_index"] == 0],
        "continuation changed the original fixed source roster",
    )
    remaining = prepared["remaining_records"]
    require(
        len(remaining) == 20
        and sum(record["native_steps"] + 40 for record in remaining) == 3652
        and prepared["config"]["new_physics_steps"] == 3652,
        "continuation remaining20 physics budget differs",
    )
    selected = [record for record in remaining if record["task_index"] == task_index]
    require(len(selected) == 10 and len({record["name"] for record in selected}) == 10, "fixed ten task cases required")
    require(
        all(43 <= record["native_steps"] <= 600 for record in selected), "native source length outside fixed budget"
    )
    return selected


def guard_receipt(path, name, phase, continuation_ref):
    actual = environment_snapshot()
    receipt = dict(
        schema_version="s2a.native-training-case-environment.v1",
        name=name,
        phase=phase,
        pid=os.getpid(),
        ppid=os.getppid(),
        argv=list(sys.argv),
        continuation_prepared_ref=continuation_ref,
        actual_environment=actual,
        bytecode=os.environ.get("PYTHONDONTWRITEBYTECODE"),
        passed=False,
    )
    try:
        strict_native_environment(actual)
        receipt["passed"] = True
    except Exception as error:
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        write_once(path, receipt)
        raise
    return write_once(path, receipt)


def run(args):
    strict_native_environment()
    require(args.seed == 42 and args.task_index in (1, 2), "fixed continuation seed/task required")
    from prepare_wam_native_training_continuation import read_prepared as read_prepared_new

    prepared_path = Path(args.prepared).resolve()
    prepared = read_prepared_new(prepared_path, args.prepared_sha256)
    require(
        prepared["schema_version"] == "s2a.native-training-continuation-preparation.v1" and prepared["seed"] == 42,
        "continuation preparation schema/seed differs",
    )
    original_ref = prepared["original_prepared_ref"]
    require(reference(original_ref["path"]) == original_ref, "original prepared source SHA changed")
    original = json.loads(Path(original_ref["path"]).read_text())
    require(original["schema_version"] == "s2a.native-training-preparation.v1", "original source schema differs")
    selected = task_records(prepared, original, args.task_index)
    destination = Path(prepared["output_root"]) / "collection" / f"task{args.task_index}"
    destination.mkdir(parents=True, exist_ok=False)
    environments = destination / "environment"
    environments.mkdir()
    continuation_ref = reference(prepared_path)
    readiness_path = destination / "environment_readiness.json"
    progress = dict(environment_initializations=0, physics_steps=0, current_canvas_images=0)
    result = dict(
        schema_version="s2a.native-training-continuation-task.v1",
        prepared_ref=original_ref,
        continuation_prepared_ref=continuation_ref,
        task_index=args.task_index,
        seed=42,
        complete=False,
        passed=False,
        cases=[],
        case_environment=[],
        native_quality_passed=False,
        cached_cases_recollected=0,
        wam_initializations=0,
        denoise_calls=0,
        fits=0,
        fresh_cases=0,
        future_rgb_frames=0,
        source_hdf5_reads=0,
        automatic_retry=False,
    )
    try:
        try:
            readiness = restore_cv2_environment(prepared["cv2_probe_ref"], prepared["cv2_runtime_source_refs"])
        except EnvironmentAdmissionError as error:
            result["environment_readiness_ref"] = write_once(readiness_path, error.receipt)
            raise
        readiness["continuation_prepared_ref"] = continuation_ref
        result["environment_readiness_ref"] = write_once(readiness_path, readiness)
        result["actual_environment"] = {
            key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTHONDONTWRITEBYTECODE")
        }
        strict_native_environment()
        write_once(
            destination / "worker_started.json",
            dict(
                pid=os.getpid(),
                ppid=os.getppid(),
                argv=list(sys.argv),
                cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
                ld=os.environ.get("LD_LIBRARY_PATH"),
                actual_environment=result["actual_environment"],
                task_index=args.task_index,
                prepared_ref=original_ref,
                continuation_prepared_ref=continuation_ref,
                environment_readiness_ref=result["environment_readiness_ref"],
            ),
        )
        for record in selected:
            name = record["name"]
            before_path, after_path = environments / f"{name}_before.json", environments / f"{name}_after.json"
            item = dict(name=name)
            result["case_environment"].append(item)
            try:
                item["before_ref"] = guard_receipt(before_path, name, "before_case", continuation_ref)
            finally:
                if before_path.exists():
                    item["before_ref"] = reference(before_path)
            read_prepared_new(prepared_path, args.prepared_sha256)
            try:
                result["cases"].append(collect_case(original, original_ref, record, destination / name, progress))
            finally:
                try:
                    item["after_ref"] = guard_receipt(after_path, name, "after_case", continuation_ref)
                finally:
                    if after_path.exists():
                        item["after_ref"] = reference(after_path)
            read_prepared_new(prepared_path, args.prepared_sha256)
        strict_native_environment()
        read_prepared_new(prepared_path, args.prepared_sha256)
        qualities = []
        for ref in result["cases"]:
            require(reference(ref["path"]) == ref, "completed case changed during continuation")
            case = json.loads(Path(ref["path"]).read_text())
            require(
                case["schema_version"] == "s2a.native-training-case.v1"
                and case["complete"] is case["passed"] is True
                and case["prepared_ref"] == original_ref,
                "continuation case lost its original source or protocol completeness",
            )
            qualities.append(case["native_quality_passed"])
        require(progress["environment_initializations"] == 10, "continuation task environment budget differs")
        require(
            progress["physics_steps"] == sum(record["native_steps"] + 40 for record in selected), "task step mismatch"
        )
        require(progress["current_canvas_images"] == 50, "continuation task image budget differs")
        result.update(complete=True, passed=True, native_quality_passed=all(qualities))
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        result["failure_environment"] = environment_snapshot()
    result.update(progress)
    write_once(destination / "result.json", result)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--task-index", required=True, type=int, choices=(1, 2))
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
