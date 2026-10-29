"""Verify an existing command branch under the migrated CPU runtime.

Default is metadata-only admission. Actual execution requires --execute and a
completed native component-QA receipt; the caller owns the actual process wait.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from s2a.native_runtime_replay import ATOL, LIMITS, REPO, ReplayFailure, run_native_replay, validate_saved_branch

CONFIG = REPO / "configs/native_runtime_replay_20261029.json"
PLAN = REPO / "plans/v0.0.1/native_runtime_replay_20261029.md"
NATIVE_PYTHON = REPO / "outputs/libero_env311/bin/python"
SOURCE_PATHS = (
    "src/s2a/native_runtime_replay.py",
    "scripts/verify_native_runtime_replay.py",
    "tests/test_native_runtime_replay.py",
    "plans/v0.0.1/native_runtime_replay_20261029.md",
    "configs/native_runtime_replay_20261029.json",
)
QA_TESTS = ("tests/test_native_runtime_replay.py",)
ADDITIONAL_RUNTIME_SOURCES = (
    "scripts/collect_wam_native_training.py",
    "src/s2a/native_training.py",
    "src/s2a/action_reference.py",
    "src/s2a/cameras.py",
    "src/s2a/checkpoint_simulation.py",
    "src/s2a/current_coupled_state.py",
    "src/s2a/gripper_motor_state.py",
)
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Absolute nontraversing source path required")
    require(not str(path).startswith("/share/project/") and path.name != "long_joint_inverse.py", "Forbidden source")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def checked(ref, refs=None):
    actual = reference(ref["path"])
    require(actual["sha256"] == ref["sha256"], f"Changed source: {ref['path']}")
    if refs is not None:
        require(ref["path"] not in refs or refs[ref["path"]] == ref["sha256"], "Conflicting source identity")
        refs[ref["path"]] = ref["sha256"]
    return Path(ref["path"])


def read(ref, refs=None):
    return json.loads(checked(ref, refs).read_text())


def closed_owner(owner, expected_ref, refs):
    require(
        owner["complete"] is owner["passed"] is True and owner["result_ref"] == expected_ref,
        "Closed exact historical owner",
    )
    command = owner["owned_command"]
    exit_record, release = read(command["exit"], refs), read(command["release"], refs)
    require(
        exit_record["actual_wait"] is True
        and release["reaped"] is True
        and exit_record["returncode"] == release["returncode"] == 0,
        "Historical actual wait/reap0",
    )


def inspect_inputs(config):
    """No simulator or model import, construction, stepping, or output writes."""
    refs = {}
    for relative in SOURCE_PATHS:
        checked(reference(REPO / relative), refs)
    require(
        config["schema"] == "s2a.native-runtime-replay-inputs.v1"
        and config["historical_branch"] == "fk_first_long_h8"
        and config["seed"] == 42
        and config["trial"] == 0
        and config["physics_budget"] == 377
        and config["native_integration_budget"] == 9425
        and config["prefix_steps"] == 368
        and config["saved_continuation_steps"] == 9
        and config["absolute_tolerance"] == ATOL
        and config["relative_tolerance"] == 0
        and config["timeout_seconds"] == 1800,
        "Fixed replay registration",
    )
    require(
        config["automatic_retry"] is config["control_qualified"] is False
        and config["world_model_calls"] == config["fresh_cases"] == config["newly_selected_commands"] == 0,
        "Historical replay only",
    )
    require(config["output_dir"] == str(REPO / "outputs/native_runtime_replay_20261029"), "Exclusive registered output")
    boot = read(config["bootstrap_ref"], refs)
    owner = read(config["bootstrap_owner_ref"], refs)
    boot_config = read(config["bootstrap_config_ref"], refs)
    require(
        boot["complete"] is boot["passed"] is boot["bootstrap_verified"] is boot["source_unchanged"] is True
        and boot["config_ref"] == config["bootstrap_config_ref"]
        and boot["control_qualified"] is boot["historical_runtime_equivalence_claimed"] is False,
        "Verified current v3 bootstrap, without historical equivalence claim",
    )
    require(
        owner["complete"] is owner["passed"] is True
        and owner["result_ref"] == config["bootstrap_ref"]
        and len(owner["commands"]) == 1,
        "Closed exact bootstrap owner",
    )
    command = owner["commands"][0]
    require(
        command["actual_wait"] is command["reaped"] is True
        and command["returncode"] == 0
        and command["signal_sent"] is command["automatic_retry"] is False,
        "Actual bootstrap wait/reap0",
    )
    for ref in boot["source_refs_after"]:
        checked(ref, refs)
    producer, checker = read(config["producer_ref"], refs), read(config["checker_ref"], refs)
    for key in ("producer", "checker"):
        historical_owner = read(config[f"{key}_owner_ref"], refs)
        closed_owner(historical_owner, config[f"{key}_ref"], refs)
        if key == "producer":
            historical_sources = read(historical_owner["registration_ref"], refs)["source_sha256"]
            for relative in ADDITIONAL_RUNTIME_SOURCES:
                path = str(REPO / relative)
                checked(dict(path=path, sha256=historical_sources[path]), refs)
    require(
        producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True,
        "Previously closed branch and independent replay",
    )
    require(
        producer["physics_steps"] == checker["physics_steps"] == 377
        and producer["continuation_steps"] == checker["continuation_steps"] == 9
        and producer["last_consumed_cursor"] == checker["last_consumed_cursor"] == 8,
        "Exact archived scope",
    )
    registration = read(producer["registration_ref"], refs)
    original_producer = read(registration["original_producer_ref"], refs)
    original_registration = read(original_producer["registration_ref"], refs)
    initial_joint = np.asarray(read(original_registration["braking_numeric_ref"], refs)["initial_joint"])
    require(initial_joint.shape == (7,) and np.isfinite(initial_joint).all(), "Original controller initial_joint")
    checker_registration = read(checker["registration_ref"], refs)
    require(
        checker_registration["producer_ref"] == config["producer_ref"] and registration["horizon"] == 8,
        "Independent check bound to this H8 branch",
    )
    require(
        producer["stopped"]["step"] == 377 and producer["stopped"]["action_dispatched"] is False,
        "Original stop retained, no cursor9 action",
    )
    original_manifest = read(registration["manifest_ref"], refs)
    manifest = read(boot_config["record_ref"], refs)
    require(manifest["record"] == original_manifest["record"], "Same historical record and initial state")
    before, after = original_manifest["runtime_config"].copy(), manifest["runtime_config"].copy()
    for key in ("libero_root", "libero_config", "osmesa_library"):
        before.pop(key)
        after.pop(key)
    require(before == after, "Only runtime locations migrated")
    commands = np.asarray(read(producer["commands_ref"], refs))
    with np.load(checked(producer["states_ref"], refs), allow_pickle=False) as archive:
        states = archive["sim_states"].copy()
    # Both historical branches already saved precisely the same endpoint bytes.
    require(
        checker["states_ref"]["sha256"] == producer["states_ref"]["sha256"], "Historical endpoint archives identical"
    )
    checked(checker["states_ref"], refs)
    periods = read(producer["periods_ref"], refs)
    validate_saved_branch(commands, states, periods)
    targets = np.asarray(read(registration["chunks_ref"], refs)[46]["effective_targets_tcp"])
    tool = np.asarray(read(registration["tool_ref"], refs)["source_from_tcp"])
    record_model = boot["bootstrap"]["model_records"][2]
    require(record_model["stage"] == "record_xml" and record_model["completed"] is True, "Closed mapped record model")
    record_xml = checked(record_model["xml_ref"], refs).read_text()
    assets = {}
    for model in boot["bootstrap"]["model_records"]:
        for asset in model["assets"]:
            ref = asset["reference"]
            checked(ref, refs)
            assets[ref["path"]] = ref["sha256"]
    return dict(
        refs=refs,
        bootstrap_config=boot_config,
        manifest=manifest,
        record_xml=record_xml,
        commands=commands,
        states=states,
        periods=periods,
        targets=targets,
        tool=tool,
        asset_refs=assets,
        controller_contract=boot["bootstrap"]["controller_contract"],
        initial_joint=initial_joint,
    )


def verify_qa(qa_ref):
    qa = read(qa_ref)
    require(
        qa["complete"] is qa["passed"] is qa["actual_wait"] is qa["reaped"] is qa["source_unchanged"] is True
        and qa["returncode"] == 0
        and qa["signal_sent"] is qa["automatic_retry"] is False,
        "Completed component QA actual wait/reap0",
    )
    require(
        qa["environment"] == CPU_ENVIRONMENT
        and qa["python_version"] == "3.11.15"
        and qa["python_executable"] == str(NATIVE_PYTHON),
        "Native CPU QA environment",
    )
    require(
        qa["command"][:4] == [str(NATIVE_PYTHON), "-m", "pytest", "-q"]
        and qa["command"][4:5] == list(QA_TESTS)
        and all(item.startswith("--basetemp=") for item in qa["command"][5:]),
        "Replay component test command",
    )
    expected = {str(REPO / relative): reference(REPO / relative)["sha256"] for relative in SOURCE_PATHS}
    require(
        qa["source_sha256"] == expected and qa["source_count"] == len(expected) and qa["test_module_count"] == 1,
        "Exact five-source component QA",
    )
    checked(qa["log_ref"])
    checked(qa["driver_ref"])
    return qa


def publish(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=convert)
        stream.write("\n")


def main(*, execute=False, qa_ref=None):
    require(
        Path.cwd() == REPO and all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()),
        "Explicit CPU-only single-thread environment and repository cwd required",
    )
    config = read(reference(CONFIG))
    inputs = inspect_inputs(config)
    if not execute:
        print(
            json.dumps(
                dict(
                    passed=True,
                    dry_run=True,
                    source_count=len(inputs["refs"]),
                    physics_steps=0,
                    runtime_loaded=False,
                    control_qualified=False,
                )
            ),
            flush=True,
        )
        return
    require(sys.executable == str(NATIVE_PYTHON), "Actual replay uses registered native Python")
    require(qa_ref is not None, "Completed component QA required")
    verify_qa(qa_ref)
    root = Path(config["output_dir"])
    require(not root.exists(), "No overwrite or automatic retry")
    root.mkdir()
    result = dict(
        complete=False,
        passed=False,
        config_ref=reference(CONFIG),
        plan_ref=reference(PLAN),
        qa_ref=qa_ref,
        environment=CPU_ENVIRONMENT,
        pid=os.getpid(),
        ppid=os.getppid(),
        source_sha256=inputs["refs"],
        limits=LIMITS,
        control_qualified=False,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
        actual_wait_claimed=False,
    )
    try:
        from s2a.native_environment import restore_cv2_environment

        guard = inputs["bootstrap_config"]["cv2_guard"]
        result["cv2_admission"] = restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
        manifest = inputs["manifest"]

        def progress(receipt):
            if receipt.get("checked_endpoints", 0) % 16 == 0:
                print(
                    json.dumps(dict(endpoint=receipt.get("current_endpoint"), counters=receipt["counters"])), flush=True
                )

        result["replay"] = run_native_replay(
            manifest["runtime_config"],
            manifest["record"],
            inputs["record_xml"],
            inputs["commands"],
            inputs["states"],
            inputs["periods"],
            inputs["targets"],
            inputs["tool"],
            root / "work",
            allowed_asset_roots=inputs["bootstrap_config"]["allowed_asset_roots"],
            asset_refs=inputs["asset_refs"],
            expected_controller=inputs["controller_contract"],
            expected_initial_joint=inputs["initial_joint"],
            progress=progress,
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        if isinstance(error, ReplayFailure):
            result["replay"] = error.receipt
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            for path, digest in inputs["refs"].items():
                checked(dict(path=path, sha256=digest))
            if "replay" in result:
                for key in ("loaded_libraries", "loaded_sources"):
                    for ref in result["replay"].get(key, []):
                        checked(ref)
            result["source_unchanged"] = True
        except BaseException as error:
            result.update(complete=False, passed=False, source_unchanged=False, source_error=repr(error))
            raise
        finally:
            publish(root / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--qa-ref")
    parser.add_argument("--qa-sha256")
    args = parser.parse_args()
    if bool(args.qa_ref) != bool(args.qa_sha256):
        parser.error("Both QA arguments are required together")
    main(execute=args.execute, qa_ref=dict(path=args.qa_ref, sha256=args.qa_sha256) if args.qa_ref else None)
