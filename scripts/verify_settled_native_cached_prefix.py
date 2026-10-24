"""Read-only 034 audit with explicit admission of the already audited cache."""

import argparse
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import verify_settled_native_training as audit
from run_settled_native_training import publish, read, read_prepared, reference, require
from run_wam_readout import owned_command

from s2a.artifacts import REPO, sha256
from s2a.native_environment import strict_native_environment

PLAN = REPO / "plans/v0.0.1/034_cached_task_audit_addendum_20261022.md"
SOURCES = (PLAN, Path(__file__).resolve(), REPO / "tests/test_settled_native_cached_prefix.py")


def historical_cases(collection):
    require(collection["complete"] is collection["passed"] is True, "original30 collection incomplete")
    cached = read(collection["cached_task_ref"])
    prefix = read(collection["prefix_verification_ref"])
    require(
        cached["task_index"] == 0
        and cached["complete"] is cached["passed"] is False
        and cached["error"]["type"] == "ValueError"
        and cached["error"]["message"] == "030 source guard requires S2A cwd and empty CUDA/LD"
        and len(cached["cases"]) == collection["cached_cases"] == 10
        and cached["physics_steps"] == collection["cached_physics_steps"] == 1328
        and cached["current_canvas_images"] == collection["cached_current_canvas_images"] == 50
        and cached["environment_initializations"] == 10
        and all(
            cached[key] == 0
            for key in ("wam_initializations", "denoise_calls", "fits", "fresh_cases", "future_rgb_frames")
        ),
        "preserved failed task0 envelope differs",
    )
    require(
        prefix["schema_version"] == "s2a.native-training-continuation-independent-verification.v1"
        and prefix["phase"] == "prefix"
        and prefix["passed"] is prefix["prefix_verified"] is prefix["evidence_integrity_passed"] is True
        and prefix["prepared"] == collection["prepared_ref"]
        and prefix["cached_case_refs"] == cached["cases"]
        and prefix["model_training_authorized"] is prefix["fresh_performance_authorized"] is False,
        "cached ten cases lack the exact independent prefix audit",
    )
    tasks = [cached] + [read(item["result"]) for item in collection["task_results"]]
    require([task["task_index"] for task in tasks] == [0, 1, 2], "original30 task roster changed")
    for task in tasks[1:]:
        require(task["passed"] is task["complete"] is True, "new continuation task incomplete")
    require(all(len(task["cases"]) == 10 for task in tasks), "original ten-case task denominator changed")
    result = [read(ref) for task in tasks for ref in task["cases"]]
    require(all(case["passed"] is case["complete"] is True for case in result), "original complete case missing")
    return result


def validate_registration(ref):
    registration = read(ref)
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        for path, digest in registration[key].items():
            require(reference(path)["sha256"] == digest, "registered audit source/input changed")
    prepared = read_prepared(registration["prepared_ref"])
    failed = read(registration["preserved_failed_owner_ref"])
    require(failed["passed"] is failed["complete"] is False, "original034 False owner changed")
    require(failed["prepared_ref"] == registration["prepared_ref"], "failed owner preparation differs")
    require(read(registration["collection_ref"])["prepared_ref"] == registration["prepared_ref"], "collection differs")
    return registration, prepared


def verify(ref):
    registration, prepared = validate_registration(ref)
    destination = Path(registration["root"]) / "review"
    destination.mkdir(exist_ok=False)
    original_publish = audit.publish

    def publish_result(path, data):
        require(Path(path) == Path(prepared["root"]) / "review/verified.json", "unexpected audit publication")
        data["cached_task_audit_addendum"] = dict(
            registration_ref=ref,
            preserved_failed_owner_ref=registration["preserved_failed_owner_ref"],
            cached_task0_original_passed=False,
            cached_task0_original_complete=False,
            new_physics_steps=0,
            original034_owner_rewritten=False,
        )
        return original_publish(destination / "verified.json", data)

    prepared_ref = registration["prepared_ref"]
    with patch.object(audit, "historical_cases", historical_cases), patch.object(audit, "publish", publish_result):
        audit.verify(argparse.Namespace(prepared=prepared_ref["path"], prepared_sha256=prepared_ref["sha256"], seed=42))
    validate_registration(ref)


def prepare():
    strict_native_environment()
    root = REPO / "outputs/wam_stability_20261021/settled_native_training"
    prepared_ref = reference(root / "preparation/prepared.json")
    prepared = read_prepared(prepared_ref)
    require(read(reference(root / "collection/result.json"))["complete"], "complete collection required")
    historical_cases(read(prepared["config"]["native_collection_ref"]))
    inputs = {
        str(root / name): sha256(root / name)
        for name in (
            "preparation/prepared.json",
            "collection/result.json",
            "orchestration/result.json",
            "orchestration/independent/command.log",
        )
    }
    destination = root / "audit_addendum"
    (destination / "snapshots").mkdir(parents=True, exist_ok=False)
    sources, snapshots = {}, {}
    for index, source in enumerate(SOURCES):
        digest = sha256(source)
        target = destination / "snapshots" / f"{index:02d}_{source.name}"
        shutil.copyfile(source, target)
        require(sha256(target) == digest, "addendum snapshot differs")
        sources[str(source)], snapshots[str(target)] = digest, digest
    return publish(
        destination / "registration.json",
        dict(
            root=str(destination),
            prepared_ref=prepared_ref,
            collection_ref=reference(root / "collection/result.json"),
            preserved_failed_owner_ref=reference(root / "orchestration/result.json"),
            source_sha256=sources,
            snapshot_sha256=snapshots,
            input_sha256=inputs,
            new_physics_steps=0,
            new_model_calls=0,
            new_fits=0,
            fresh_cases=0,
        ),
    )


def run(ref):
    registration, _ = validate_registration(ref)
    root = Path(registration["root"])
    command = [
        str(REPO / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--stage",
        "verify",
        "--registration",
        ref["path"],
        "--registration-sha256",
        ref["sha256"],
    ]
    result = dict(
        passed=False, complete=False, registration_ref=ref, new_physics_steps=0, new_model_calls=0, new_fits=0
    )
    try:
        result["owned_command"] = owned_command(
            command,
            dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
            root / "orchestration",
            timeout=1800,
        )
        verified = reference(root / "review/verified.json")
        require(read(verified)["complete"] and read(verified)["passed"], "addendum audit incomplete")
        validate_registration(ref)
        result.update(passed=True, complete=True, verified_ref=verified)
    finally:
        publish(root / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run", "verify"), required=True)
    parser.add_argument("--registration")
    parser.add_argument("--registration-sha256")
    args = parser.parse_args()
    strict_native_environment()
    if args.stage == "prepare":
        print(json.dumps(prepare()))
    else:
        ref = dict(path=str(Path(args.registration).resolve()), sha256=args.registration_sha256)
        (run if args.stage == "run" else verify)(ref)


if __name__ == "__main__":
    main()
