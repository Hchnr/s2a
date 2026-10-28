"""Prepare source-bound runtime files without importing or running the simulator."""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path

from s2a.native_runtime_path_snapshot import (
    OLD_SOURCE_ROOT,
    REPO,
    SAFE_SOURCE_ROOT,
    inspect_runtime_sources,
    prepare_runtime_snapshot,
    verify_runtime_snapshot,
)

CONFIG = REPO / "configs/native_runtime_path_snapshot_20261029.json"
PLAN = REPO / "plans/v0.0.1/native_runtime_path_snapshot_20261029.md"
RESULT = REPO / "outputs/native_runtime_path_snapshot_preparation_20261029/result.json"
NATIVE_PYTHON = REPO / "outputs/libero_env311/bin/python"
CPU_ENVIRONMENT = dict(
    CUDA_VISIBLE_DEVICES="",
    LD_LIBRARY_PATH="",
    PYTHONDONTWRITEBYTECODE="1",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    LP_NUM_THREADS="1",
)
QA_TESTS = (
    "tests/test_native_runtime_path_snapshot.py",
    "tests/test_native_runtime_path_preparation.py",
)
SOURCE_PATHS = (
    "src/s2a/native_runtime_path_snapshot.py",
    "scripts/prepare_native_runtime_path_snapshot.py",
    "plans/v0.0.1/native_runtime_path_snapshot_20261029.md",
    "configs/native_runtime_path_snapshot_20261029.json",
    *QA_TESTS,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def repository_output(path):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(REPO / "outputs"), "Output must stay in repository outputs")
    relative = path.relative_to(REPO)
    require(".." not in relative.parts, "Output traversal is forbidden")
    current = REPO
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), "Output ancestors cannot be symlinks")
    require(path.resolve().is_relative_to((REPO / "outputs").resolve()), "Resolved output escapes repository")
    return path


def reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def checked(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact file reference")
    path = Path(ref["path"])
    require(path.is_absolute() and reference(path) == ref, "File path/SHA mismatch")
    return path


def read(ref):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value

    def reject(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    def number(token):
        value = float(token)
        require(math.isfinite(value), "Finite JSON number required")
        return value

    data = checked(ref).read_bytes()
    require(hashlib.sha256(data).hexdigest() == ref["sha256"], "JSON changed while reading")
    return json.loads(data, object_pairs_hook=unique, parse_constant=reject, parse_float=number)


def verify_native_qa(qa_ref):
    qa = read(qa_ref)
    require(
        qa["complete"] is qa["passed"] is qa["actual_wait"] is True
        and qa["returncode"] == 0
        and qa["signal_sent"] is False
        and qa["automatic_retry"] is False
        and qa["environment"] == CPU_ENVIRONMENT
        and qa["python_version"].startswith("3.11.")
        and Path(qa["python_executable"]).resolve() == NATIVE_PYTHON.resolve(),
        "Completed native CPU QA with actual successful wait and no retry",
    )
    require(set(qa["source_sha256"]) == set(SOURCE_PATHS), "Exact final preparation sources and tests")
    command = qa["command"]
    require(
        isinstance(command, list)
        and len(command) == 7
        and all(isinstance(token, str) for token in command)
        and command[:6] == [str(NATIVE_PYTHON), "-m", "pytest", "-q", *QA_TESTS]
        and command[6].startswith("--basetemp="),
        "Both meaningful test modules executed with the exact native pytest argv",
    )
    repository_output(command[6].removeprefix("--basetemp="))
    for path, digest in qa["source_sha256"].items():
        checked(dict(path=str(REPO / path), sha256=digest))
    checked(qa["log_ref"])
    return qa


def inspect_inputs(config):
    require(config["schema"] == "s2a.native-runtime-path-snapshot-inputs.v1", "Registered input schema")
    require(config["max_files"] == 10000 and config["max_bytes"] == 2**30, "Fixed file preparation budget")
    require(
        Path(config["output_dir"]) == REPO / "outputs/native_runtime_path_snapshot_20261029",
        "One fixed repository snapshot destination",
    )
    original = read(config["original_manifest_ref"])
    runtime = original["runtime_config"]
    require(runtime["task_suite"] == "libero_goal" and runtime["task_id"] == 8, "Original bowl/plate task")
    require(runtime["seed"] == 42 and runtime["refresh_controller_after_restore"] is True, "Original seed and restore")
    refs = [config[key] for key in ("original_manifest_ref", "original_config_ref", "osmesa_ref")]
    counts = Counter()
    for declared, digest in original["simulator_source_sha256"].items():
        path = Path(declared)
        if path.is_relative_to(Path(OLD_SOURCE_ROOT) / "LIBERO/libero"):
            path = Path(SAFE_SOURCE_ROOT) / path.relative_to(OLD_SOURCE_ROOT)
            counts["libero"] += 1
        else:
            require(
                path.is_relative_to(REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite"),
                "Only registered LIBERO and repository robosuite source paths",
            )
            counts["robosuite"] += 1
        ref = dict(path=str(path), sha256=digest)
        checked(ref)
        refs.append(ref)
    require(
        counts == {"libero": 45, "robosuite": 120}
        and config["expected_historical_libero_sources"] == 45
        and config["expected_historical_robosuite_sources"] == 120,
        "All historical simulator source identities retained",
    )
    support = config["current_only_refs"]
    require(
        Counter(item["role"] for item in support) == {"system_library": 18, "task_bddl": 1, "default_osc_json": 1}
        and all(item["identity"] == "current_only" for item in support)
        and len({item["ref"]["path"] for item in support}) == 20,
        "Twenty distinct current-only dependencies; never historical qualification",
    )
    for item in support:
        path = checked(item["ref"])
        if item["role"] == "system_library":
            require(path.parent == Path("/usr/lib/x86_64-linux-gnu"), "Declared current system-library candidates")
        elif item["role"] == "task_bddl":
            require(
                path
                == Path(SAFE_SOURCE_ROOT)
                / "LIBERO/libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl",
                "Exact task BDDL",
            )
        else:
            require(
                path
                == REPO
                / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/controllers/config/osc_pose.json",
                "Default OSC JSON remains referenced at its actual installed location",
            )
        refs.append(item["ref"])
    inspected = inspect_runtime_sources(
        config["original_manifest_ref"],
        config["original_config_ref"],
        config["osmesa_ref"],
        max_files=config["max_files"],
        max_bytes=config["max_bytes"],
    )
    require(
        inspected["historical_libero_file_count"] == 45 and inspected["license_included"],
        "Complete package and license",
    )
    for ref in refs:
        checked(ref)
    return inspected, refs


def main(qa_ref, dry=False):
    require(Path.cwd() == REPO, "Run from the designated repository")
    require({key: os.environ.get(key) for key in CPU_ENVIRONMENT} == CPU_ENVIRONMENT, "CPU-only environment")
    require(Path(sys.executable).resolve() == NATIVE_PYTHON.resolve(), "Formal repository native Python")
    repository_output(RESULT)
    require(not RESULT.parent.exists(), "One preparation result; preserve failures without retry")
    qa = verify_native_qa(qa_ref)
    config_ref = dict(
        path=str(CONFIG), sha256=qa["source_sha256"]["configs/native_runtime_path_snapshot_20261029.json"]
    )
    config = read(config_ref)
    repository_output(config["output_dir"])
    require(not Path(config["output_dir"]).exists(), "One exclusive snapshot destination")
    inspected, refs = inspect_inputs(config)
    refs.extend(
        [
            config_ref,
            qa_ref,
            qa["log_ref"],
            *[dict(path=str(REPO / path), sha256=qa["source_sha256"][path]) for path in SOURCE_PATHS],
        ]
    )
    for ref in refs:
        checked(ref)
    summary = {key: inspected[key] for key in ("copied_file_count", "copied_bytes", "historical_libero_file_count")}
    if dry:
        print(json.dumps(dict(dry_run=True, passed=True, **summary, new_responses=0, physics_steps=0)), flush=True)
        return
    RESULT.parent.mkdir(parents=True, exist_ok=False)
    repository_output(RESULT)
    result = dict(
        complete=False,
        passed=False,
        native_qa_ref=qa_ref,
        config_ref=config_ref,
        plan_ref=dict(
            path=str(PLAN), sha256=qa["source_sha256"]["plans/v0.0.1/native_runtime_path_snapshot_20261029.md"]
        ),
        source_refs_before=refs,
        current_only_refs=config["current_only_refs"],
        environment=CPU_ENVIRONMENT,
        python_version=platform.python_version(),
        runtime_loaded=False,
        bootstrap_verified=False,
        control_qualified=False,
        response_evaluations=0,
        numerical_inner_steps=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        snapshot_ref = prepare_runtime_snapshot(
            config["original_manifest_ref"],
            config["original_config_ref"],
            config["osmesa_ref"],
            config["output_dir"],
            max_files=config["max_files"],
            max_bytes=config["max_bytes"],
        )
        snapshot = verify_runtime_snapshot(snapshot_ref)
        require(
            snapshot["file_snapshot_verified"] is True
            and snapshot["runtime_loaded"] is snapshot["bootstrap_verified"] is snapshot["control_qualified"] is False,
            "File preparation alone grants no runtime, bootstrap or control qualification",
        )
        for ref in refs:
            checked(ref)
        result.update(complete=True, passed=True, snapshot_ref=snapshot_ref, source_refs_unchanged=refs, **summary)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        repository_output(RESULT)
        with RESULT.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(
            json.dumps(dict(result_ref=reference(RESULT), complete=result["complete"], passed=result["passed"])),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-ref", type=Path, required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256), args.dry)
