"""Source-bound CPU bootstrap of the snapshotted native runtime, without control."""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from prepare_native_runtime_path_snapshot import checked, read, reference

from s2a.native_environment import EnvironmentAdmissionError, restore_cv2_environment, strict_native_environment
from s2a.native_runtime_path_snapshot import REPO, verify_runtime_snapshot

CONFIG = REPO / "configs/native_runtime_bootstrap_20261029.json"
PLAN = REPO / "plans/v0.0.1/native_runtime_bootstrap_20261029.md"
RESULT = REPO / "outputs/native_runtime_bootstrap_20261029/result.json"
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
    "tests/test_native_runtime_bootstrap.py",
    "tests/test_native_runtime_bootstrap_execution.py",
    "tests/test_native_runtime_path_snapshot.py",
    "tests/test_native_runtime_path_preparation.py",
    "tests/test_native_environment.py",
)
SOURCE_PATHS = (
    "src/s2a/native_runtime_bootstrap.py",
    "scripts/verify_native_runtime_bootstrap.py",
    "plans/v0.0.1/native_runtime_bootstrap_20261029.md",
    "configs/native_runtime_bootstrap_20261029.json",
    "src/s2a/native_runtime_path_snapshot.py",
    "scripts/prepare_native_runtime_path_snapshot.py",
    "src/s2a/native_environment.py",
    "src/s2a/artifacts.py",
    "src/s2a/libero_simulation.py",
    "src/s2a/geometry.py",
    "src/s2a/gripper.py",
    "src/s2a/osc.py",
    *QA_TESTS,
)
QA_SOURCES = SOURCE_PATHS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def repository_output(path):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(REPO / "outputs"), "Output stays in repository outputs")
    require(".." not in path.parts, "No output path traversal")
    current = REPO
    for part in path.relative_to(REPO).parts:
        current /= part
        require(not current.is_symlink(), "No output symlink components")
    require(path.resolve() == path, "Canonical output path")
    return path


def verify_native_qa(qa_ref):
    qa = read(qa_ref)
    require(
        qa["complete"] is qa["passed"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["source_unchanged"] is qa["driver_unchanged"] is True
        and qa["pytest_disable_plugin_autoload"] == "1"
        and qa["returncode"] == 0
        and qa["signal_sent"] is False
        and qa["automatic_retry"] is False
        and qa["environment"] == CPU_ENVIRONMENT
        and qa["python_version"].startswith("3.11.")
        and Path(qa["python_executable"]).resolve() == NATIVE_PYTHON.resolve(),
        "Completed native CPU QA with actual successful wait and no retry",
    )
    require(set(qa["source_sha256"]) == set(SOURCE_PATHS), "Exact bootstrap dependency and test sources")
    command = qa["command"]
    prefix = [str(NATIVE_PYTHON), "-m", "pytest", "-q", *QA_TESTS]
    require(
        isinstance(command, list)
        and command[:-1] == prefix
        and len(command) == len(prefix) + 1
        and isinstance(command[-1], str)
        and command[-1].startswith("--basetemp="),
        "Every registered bootstrap test module uses the exact native pytest argv",
    )
    repository_output(command[-1].removeprefix("--basetemp="))
    for path, sha in qa["source_sha256"].items():
        checked(dict(path=str(REPO / path), sha256=sha))
    checked(qa["log_ref"])
    checked(qa["driver_ref"])
    require(qa["source_count"] == len(SOURCE_PATHS) and qa["test_module_count"] == len(QA_TESTS), "QA roster counts")
    return qa


def add_ref(ref, refs):
    require(
        not ref["path"].startswith("/share/project/") and ".." not in Path(ref["path"]).parts,
        "No retired or traversing input path",
    )
    checked(ref)
    previous = refs.setdefault(ref["path"], ref)
    require(previous == ref, "No source path may carry conflicting digests")
    return ref


def finish(refs):
    for ref in refs.values():
        checked(ref)


LIMIT_NAMES = dict(
    model_build="model_construct",
    osmesa_context="context",
    mj_forward="forward",
    mj_step1="step1",
    mj_resetData="reset",
    render="render",
    read_pixels="read_pixels",
    control_step="control_step",
    mj_step="mj_step",
    mj_step2="mj_step2",
)


def collect_refs(value, refs):
    """Read references, never provenance-only path strings or historical roots."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            add_ref(dict(path=value["path"], sha256=value["sha256"]), refs)
        else:
            for item in value.values():
                collect_refs(item, refs)
    elif isinstance(value, list):
        for item in value:
            collect_refs(item, refs)


def inspect_inputs(config):
    """File and provenance admission only: no cv2, simulator, dataset or model."""
    from prepare_native_runtime_path_snapshot import inspect_inputs as inspect_snapshot_inputs

    from s2a.native_runtime_bootstrap import LIMITS

    require(config["schema"] == "s2a.native-runtime-bootstrap-inputs.v1", "Bootstrap input schema")
    require(
        config["timeout_seconds"] == 600
        and config["seed"] == 42
        and config["trial"] == 0
        and config["render"] == dict(camera_name="agentview", depth=False, height=256, width=256)
        and config["automatic_retry"]
        is config["control_qualified"]
        is config["historical_runtime_equivalence_claimed"]
        is False
        and all(
            type(config[key]) is int and config[key] == 0
            for key in ("actual_consumed_reference_count", "fresh_cases", "s2a_commands", "world_model_calls")
        ),
        "Fixed bootstrap scope, render, seed, budget and qualification",
    )
    require(set(config["limits"]) == set(LIMIT_NAMES), "Exact counter budget names")
    require(
        {LIMIT_NAMES[key]: value for key, value in config["limits"].items()}
        == {key: value for key, value in LIMITS.items() if key != "configure"},
        "Registered bootstrap counter limits",
    )
    require(repository_output(config["output_dir"]) == RESULT.parent, "Fixed bootstrap result directory")
    refs = {}
    for key in (
        "snapshot_ref",
        "snapshot_crosscheck_ref",
        "snapshot_crosscheck_execution_ref",
        "snapshot_inputs_ref",
        "record_ref",
        "original_manifest_ref",
        "controller_ref",
    ):
        add_ref(config[key], refs)
    cross = read(config["snapshot_crosscheck_ref"])
    execution = read(config["snapshot_crosscheck_execution_ref"])
    require(
        cross["complete"] is cross["passed"] is cross["refs_rechecked_unchanged"] is True
        and cross["input_manifest_ref"] == config["snapshot_ref"]
        and cross["historical_libero_sources"] == 45
        and cross["historical_robosuite_sources"] == 120
        and cross["outer_path_changes"] == 3
        and cross["inner_path_changes"] == 5
        and cross["inverse_config_object_equal"]
        is cross["inverse_manifest_object_equal"]
        is cross["original_manifest_and_config_byte_equal"]
        is True,
        "Complete independent file crosscheck bound to this snapshot",
    )
    require(
        execution["passed"] is execution["actual_wait"] is execution["reaped"] is execution["driver_unchanged"] is True
        and execution["returncode"] == 0
        and execution["signal_sent"] is execution["automatic_retry"] is False
        and execution["result_ref"] == config["snapshot_crosscheck_ref"]
        and execution["driver_ref"] == cross["driver_ref"]
        and execution["command"] == [str(NATIVE_PYTHON), cross["driver_ref"]["path"]],
        "Independent crosscheck actual wait/reap and exact result binding",
    )
    collect_refs(execution, refs)
    for key in ("driver_ref", "preparation_receipt_ref", "registered_source_refs_unchanged"):
        collect_refs(cross[key], refs)
    snapshot = verify_runtime_snapshot(config["snapshot_ref"])
    require(
        snapshot["file_snapshot_verified"] is True
        and snapshot["runtime_loaded"] is snapshot["bootstrap_verified"] is snapshot["control_qualified"] is False
        and snapshot["runtime_manifest_ref"] == config["record_ref"]
        and snapshot["original_manifest_ref"] == config["original_manifest_ref"],
        "Verified file snapshot bound to the original and migrated manifests",
    )
    snapshot_inputs = read(config["snapshot_inputs_ref"])
    require(
        snapshot_inputs["original_manifest_ref"] == config["original_manifest_ref"], "Snapshot input manifest binding"
    )
    _, original_refs = inspect_snapshot_inputs(snapshot_inputs)
    collect_refs(original_refs, refs)
    for key in (
        "runtime_config_ref",
        "original_manifest_snapshot_ref",
        "original_config_snapshot_ref",
        "source_inventory_ref",
    ):
        add_ref(snapshot[key], refs)
    for item in snapshot["file_inventory"]:
        add_ref(item["source_ref"], refs)
        add_ref(item["snapshot_ref"], refs)
    manifest = read(config["record_ref"])
    original = read(config["original_manifest_ref"])
    require(manifest["record"] == original["record"], "Bootstrap record preserved completely")
    runtime = manifest["runtime_config"]
    require(
        runtime["task_suite"] == "libero_goal" and runtime["task_id"] == 8 and runtime["seed"] == 42,
        "Original bootstrap task and seed",
    )
    expected_roots = [
        str(Path(runtime["libero_root"]) / "libero/libero/assets"),
        str(REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/models/assets"),
    ]
    require(config["allowed_asset_roots"] == expected_roots, "Exact snapshot and native robosuite asset roots")
    for path in expected_roots:
        require(Path(path).is_dir() and Path(path).resolve() == Path(path), "Asset roots are present and canonical")
    guard = config["cv2_guard"]
    require(guard["helper_ref"] == reference(REPO / "src/s2a/native_environment.py"), "Frozen cv2 helper identity")
    collect_refs(guard, refs)
    controller = read(config["controller_ref"])["contract"]
    return dict(
        manifest=manifest,
        controller=controller,
        snapshot=snapshot,
        refs=refs,
        current_only_dependencies=snapshot_inputs["current_only_refs"],
    )


def validate_bootstrap(receipt, controller):
    """Execution completion is separate from control or historical equivalence."""
    from s2a.native_runtime_bootstrap import LIMITS

    require(
        receipt["complete"]
        is receipt["passed"]
        is receipt["runtime_loaded"]
        is receipt["bootstrap_verified"]
        is receipt["cleanup_complete"]
        is True,
        "Complete bootstrap and environment cleanup",
    )
    require(
        receipt["control_qualified"] is False and receipt["limits"] == LIMITS, "Bootstrap scope and original limits"
    )
    require(receipt["loaded_sources_unchanged"] is True, "Loaded runtime sources unchanged")
    require(set(receipt["counters"]) == set(LIMITS), "Complete actual counter roster")
    for name, limit in LIMITS.items():
        item = receipt["counters"][name]
        require(
            type(item["attempted"]) is type(item["completed"]) is int
            and 0 <= item["completed"] <= item["attempted"] <= limit,
            "Actual counters stay within registered limits",
        )
    for name, count in dict(
        configure=1, model_construct=3, context=3, step1=2, reset=1, render=1, read_pixels=1
    ).items():
        require(receipt["counters"][name] == dict(attempted=count, completed=count), "Complete bootstrap calls")
    require(
        all(
            receipt[key] == 0
            for key in (
                "new_s2a_actions",
                "new_physics_steps",
                "new_world_model_calls",
                "new_fresh_episodes",
                "automatic_retries",
            )
        ),
        "No control, world model, fresh episode or retry",
    )
    require(
        all(
            key in receipt["controller_contract"] and receipt["controller_contract"][key] == value
            for key, value in controller.items()
        ),
        "Original controller contract exactly preserved",
    )
    require(
        receipt["initial_state_check"]["exact"] is True and receipt["renderer"]["backend"] == "osmesa",
        "Exact original initial state and CPU OSMesa",
    )
    frame = receipt["frame"]
    require(
        type(frame["pixel_min"]) is type(frame["pixel_max"]) is int
        and 0 <= frame["pixel_min"] < frame["pixel_max"] <= 255,
        "Nonconstant uint8 rendered frame",
    )


def main(qa_ref, dry=False):
    require(Path.cwd() == REPO, "Run from the designated repository")
    require({key: os.environ.get(key) for key in CPU_ENVIRONMENT} == CPU_ENVIRONMENT, "CPU-only environment")
    require(
        Path(sys.executable).resolve() == NATIVE_PYTHON.resolve() and platform.python_version().startswith("3.11."),
        "Formal native Python 3.11",
    )
    repository_output(RESULT)
    require(not RESULT.parent.exists(), "One exclusive bootstrap result; no overwrite or retry")
    qa = verify_native_qa(qa_ref)
    config_ref = dict(path=str(CONFIG), sha256=qa["source_sha256"]["configs/native_runtime_bootstrap_20261029.json"])
    config = read(config_ref)
    inputs = inspect_inputs(config)
    refs = inputs["refs"]
    collect_refs([qa_ref, qa["log_ref"], qa["driver_ref"], config_ref], refs)
    for path, digest in qa["source_sha256"].items():
        add_ref(dict(path=str(REPO / path), sha256=digest), refs)
    finish(refs)
    if dry:
        summary = dict(
            dry_run=True,
            passed=True,
            source_count=len(refs),
            runtime_loaded=False,
            bootstrap_verified=False,
            environment_initializations=0,
            new_responses=0,
            physics_steps=0,
            control_qualified=False,
        )
        print(json.dumps(summary), flush=True)
        return summary
    RESULT.parent.mkdir(parents=False, exist_ok=False)
    result = dict(
        schema_version="s2a.native-runtime-bootstrap-execution.v1",
        complete=False,
        passed=False,
        runtime_loaded=False,
        bootstrap_verified=False,
        control_qualified=False,
        historical_runtime_equivalence_claimed=False,
        automatic_retry=False,
        config_ref=config_ref,
        plan_ref=reference(PLAN),
        qa_ref=qa_ref,
        qa_log_ref=qa["log_ref"],
        environment=dict(CPU_ENVIRONMENT),
        pid=os.getpid(),
        ppid=os.getppid(),
        snapshot_ref=config["snapshot_ref"],
        snapshot_crosscheck_ref=config["snapshot_crosscheck_ref"],
        source_refs_before=list(refs.values()),
        source_refs_unchanged=[],
        source_unchanged=False,
        current_only_dependencies=inputs["current_only_dependencies"],
        actual_consumed_reference_count=0,
        s2a_commands=0,
        world_model_calls=0,
        fresh_cases=0,
    )
    stage = "cv2_environment_admission"
    primary_error = None
    try:
        guard = config["cv2_guard"]
        result["cv2_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"], expected_cwd=REPO
        )
        strict_native_environment(expected_cwd=REPO)
        require(
            {key: os.environ.get(key) for key in CPU_ENVIRONMENT} == CPU_ENVIRONMENT,
            "CPU environment unchanged after cv2",
        )
        stage = "native_bootstrap"
        from s2a.native_runtime_bootstrap import run_native_bootstrap

        result["bootstrap"] = run_native_bootstrap(
            inputs["manifest"]["runtime_config"],
            inputs["manifest"]["record"],
            RESULT.parent / "work",
            allowed_asset_roots=config["allowed_asset_roots"],
        )
        stage = "bootstrap_contract"
        strict_native_environment(expected_cwd=REPO)
        require(
            {key: os.environ.get(key) for key in CPU_ENVIRONMENT} == CPU_ENVIRONMENT,
            "CPU environment unchanged after bootstrap",
        )
        validate_bootstrap(result["bootstrap"], inputs["controller"])
        stage = "source_closure"
        collect_refs(result["bootstrap"], refs)
        finish(refs)
        result.update(
            complete=True,
            passed=True,
            runtime_loaded=True,
            bootstrap_verified=True,
            source_refs_unchanged=list(refs.values()),
            source_unchanged=True,
        )
    except Exception as error:
        primary_error = error
        from s2a.native_runtime_bootstrap import BootstrapFailure

        if isinstance(error, EnvironmentAdmissionError):
            result["cv2_admission"] = error.receipt
        if isinstance(error, BootstrapFailure):
            result["bootstrap"] = error.receipt
        result["runtime_loaded"] = bool(result.get("bootstrap", {}).get("runtime_loaded", False))
        result["failure_stage"] = result.get("bootstrap", {}).get("failure_stage", stage)
        result["error"] = dict(type=type(error).__name__, message=str(error))
        try:
            collect_refs(result.get("bootstrap", {}), refs)
            finish(refs)
            result["source_refs_unchanged"] = list(refs.values())
            result["source_unchanged"] = True
        except Exception as source_error:
            result["source_closure_error"] = dict(type=type(source_error).__name__, message=str(source_error))
        raise
    finally:
        result["source_refs_after"] = list(refs.values())
        result["source_count"] = len(refs)
        try:
            with RESULT.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(
                json.dumps(dict(result_ref=reference(RESULT), complete=result["complete"], passed=result["passed"])),
                flush=True,
            )
        except Exception as publication_error:
            if primary_error is None:
                raise
            # The original exception remains in flight. Publication may itself
            # be unavailable, so retain the secondary failure in its traceback.
            result["publication_error"] = dict(type=type(publication_error).__name__, message=str(publication_error))
            primary_error.add_note(
                f"Bootstrap result publication also failed: {type(publication_error).__name__}: {publication_error}"
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-ref", required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(dict(path=args.qa_ref, sha256=args.qa_sha256), dry=args.dry)
