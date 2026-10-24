"""Prepare and own the finite, CPU-only saved-state terminal diagnostic.

Admission hashes only the four registered caches and safe source/JSON inputs.
It never opens an HDF, image, expert command file, or model checkpoint.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

from run_wam_readout import owned_command

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/native_terminal_predicate_diagnostic_20261021.json"
CONFIG_SHA = "0dc500ff423a7ade96b537a30d0945a2fc62b36ce228259637739189e68c7992"
PLAN = REPO / "plans/v0.0.1/030_native_terminal_predicates_20261021.md"
PLAN_SHA = "e733e9b5c78c394c6d2bfbc989c1b23288a4e5893cd7f347aae6a085bc5c7c27"
NEW_SOURCES = (
    "plans/v0.0.1/030_native_terminal_predicates_20261021.md",
    "configs/native_terminal_predicate_diagnostic_20261021.json",
    "scripts/run_native_terminal_predicate_diagnostic.py",
    "tests/test_native_terminal_predicate_runner.py",
    "scripts/collect_native_terminal_predicate_diagnostic.py",
    "tests/test_native_terminal_predicate_collection.py",
    "scripts/verify_native_terminal_predicate_diagnostic.py",
    "tests/test_native_terminal_predicate_verification.py",
)
FIXED_CASES = (
    ("task1_native_demo_1", "failure", [174, 175, 176, 177]),
    ("task1_native_demo_35", "failure", [158, 159, 160, 161]),
    ("task1_native_demo_12", "matched_positive", [177, 178, 179, 180]),
    ("task1_native_demo_13", "matched_positive", [155, 156, 157, 158]),
)
FALSE_FLAGS = (
    "original_native_quality_passed",
    "eligible_for_model_training_registration",
    "model_training_authorized",
    "fresh_performance_authorized",
    "automatic_retry",
)
ZERO_BUDGET = (
    "mj_step_calls",
    "mj_step2_calls",
    "static_mj_step1_calls",
    "integrated_physics_steps",
    "source_hdf5_reads",
    "render_calls",
    "current_canvases",
    "wam_initializations",
    "denoise_calls",
    "fits",
    "fresh_cases",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def cpu_guard():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == ""
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "static diagnostic requires repository cwd and explicit empty CUDA/LD/no bytecode",
    )


def check(path, digest, inputs):
    path = Path(path).resolve()
    require(
        path.suffix not in (".hdf5", ".h5", ".png", ".jpg", ".npy", ".pt", ".pth", ".safetensors"),
        "static diagnostic refuses HDF/image/expert/weight content reads",
    )
    actual = sha256(path)
    require(digest is None or actual == digest, f"static diagnostic input SHA changed: {path}")
    require(str(path) not in inputs or inputs[str(path)] == actual, f"static diagnostic input mutated: {path}")
    inputs[str(path)] = actual
    return path


def read_ref(ref, inputs):
    require(set(ref) == {"path", "sha256"}, "invalid static diagnostic reference")
    return json.loads(check(ref["path"], ref["sha256"], inputs).read_text())


def budget_guard(config):
    require(
        config["seed"] == 42
        and config["reset_contract"] == "single_delegate_no_retry_process_local"
        and all(config[key] is False for key in FALSE_FLAGS),
        "registered static diagnostic seed/reset/permissions changed",
    )
    require(
        config["cases"] == [dict(name=n, role=r, frame_indices=f) for n, r, f in FIXED_CASES],
        "fixed four cases / sixteen endpoints changed",
    )
    expected = dict.fromkeys(ZERO_BUDGET, 0)
    expected.update(
        environment_initializations=4,
        installed_saved_endpoints=16,
        static_official_success_calls=16,
        setup_official_success_calls_max=16,
        total_official_success_calls_max=32,
        additional_atomic_predicate_calls_max=64,
        setup_mj_step1_calls_max=64,
    )
    require(config["budget"] == expected, "static diagnostic registered budget changed")


def predicate_inventory(config):
    root = Path(config["predicate_runtime_root"]).resolve()
    require(config["predicate_runtime_extensions"] == [".py", ".bddl"], "predicate source inventory extensions changed")
    paths = subprocess.check_output(["rg", "--files", "--hidden", "--no-ignore", str(root)], text=True).splitlines()
    selected = sorted(Path(p).resolve() for p in paths if Path(p).suffix in (".py", ".bddl"))
    require(selected and all(p.is_relative_to(root) for p in selected), "empty or escaped predicate source inventory")
    return {str(path): sha256(path) for path in selected}


def lifecycle_guard(lifecycle, inputs):
    launch, end, release = (read_ref(lifecycle[key], inputs) for key in ("launch", "exit", "release"))
    require(
        end.get("actual_wait") is True
        and end.get("returncode") == 0
        and release.get("reaped") is True
        and release.get("signal_sent") is False
        and release.get("returncode") == 0
        and launch.get("automatic_retry") is False
        and launch.get("selected_environment") == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        "owned CPU child must normally wait/reap without signal or retry",
    )


def build_metadata():
    """Rebuild the closed input manifest without importing or loading a simulator."""
    inputs = {}
    config = read_ref(dict(path=str(CONFIG), sha256=CONFIG_SHA), inputs)
    check(PLAN, PLAN_SHA, inputs)
    budget_guard(config)
    refs = {key: config[key] for key in config if key.startswith("native_") and key.endswith("_ref")}
    documents = {key: read_ref(ref, inputs) for key, ref in refs.items()}
    native, collection, audit, owner, terminal = (
        documents[key]
        for key in (
            "native_prepared_ref",
            "native_collection_ref",
            "native_verification_ref",
            "native_owner_ref",
            "native_tool_exit_ref",
        )
    )
    require(
        native["schema_version"] == "s2a.native-training-continuation-preparation.v1"
        and len(native["source_sha256"]) == len(native["snapshot_sha256"]) == 423
        and len(native["runtime_source_refs"]) == 12
        and len(native["records"]) == 30,
        "complete original 423-source / 30-case native preparation required",
    )
    require(
        audit["passed"] is audit["collection_verified"] is audit["evidence_integrity_passed"] is True
        and audit["prepared"] == refs["native_prepared_ref"]
        and audit["native_quality"]
        == dict(
            total_cases=30,
            stable_native_cases=28,
            stable_native_by_task={"0": 10, "1": 8, "2": 10},
            failed_native_cases=["task1_native_demo_1", "task1_native_demo_35"],
            eligible_for_model_training_registration=False,
        )
        and audit["eligible_for_model_training_registration"] is False
        and owner["complete"] is owner["passed"] is collection["complete"] is collection["passed"] is True
        and owner["producer_result"] == refs["native_collection_ref"]
        and owner["verification_ref"] == refs["native_verification_ref"]
        and terminal["actual_tool_terminal_exit_code"] == 0
        and terminal["owner_original_birth_live"] is terminal["root_manual_popen_wait_claimed"] is False
        and terminal["owner_result_ref"] == refs["native_owner_ref"]
        and len(owner["owned_commands"]) == len(terminal["all_four_actual_owned_children"]) == 4,
        "complete independently audited native28/30 and terminal ownership evidence required",
    )
    for doc in (native, collection, audit, owner, terminal):
        require(
            doc["model_training_authorized"] is doc["fresh_performance_authorized"] is False,
            "native evidence must not authorize training/fresh",
        )
    for lifecycle in owner["owned_commands"]:
        lifecycle_guard(lifecycle, inputs)
    for field in ("source_sha256", "snapshot_sha256", "first_continuation_snapshot_sha256"):
        for path, digest in native[field].items():
            check(path, digest, inputs)
    for ref in native["runtime_source_refs"]:
        check(ref["path"], ref["sha256"], inputs)
    read_ref(native["cv2_probe_ref"], inputs)
    extra_runtime = [reference(REPO / path) for path in config["extra_runtime_source_paths"]]
    require(len(extra_runtime) == 12, "registered contact/setup/MuJoCo runtime identities incomplete")
    for ref in extra_runtime:
        check(ref["path"], ref["sha256"], inputs)
    inventory = predicate_inventory(config)
    require(
        Path(config["predicate_runtime_root"])
        .resolve()
        .is_relative_to(Path(native["policy_config"]["libero_root"]).resolve()),
        "predicate inventory must use the original LIBERO package",
    )
    for path, digest in inventory.items():
        check(path, digest, inputs)
    task_ref = next(item["result"] for item in collection["task_results"] if item["task_index"] == 1)
    task = read_ref(task_ref, inputs)
    require(task["complete"] is task["passed"] is True and task["task_index"] == 1, "wine task cache incomplete")
    case_refs = {Path(ref["path"]).parent.name: ref for ref in task["cases"]}
    original_records = {r["name"]: r for r in native["records"]}
    gold = {r["name"]: r for r in audit["case_verification"]}
    cases = []
    for name, role, frames in FIXED_CASES:
        ref = case_refs[name]
        case = read_ref(ref, inputs)
        record = original_records[name]
        require(
            case["complete"] is case["passed"] is True
            and case["record"] == record
            and record["task_index"] == 1
            and record["native_steps"] == frames[0]
            and case["prepared_ref"] == native["original_prepared_ref"],
            "registered case identity/initial state/native endpoint differs",
        )
        for key in ("arrays", "endpoint_receipts"):
            check(case[key]["path"], case[key]["sha256"], inputs)
        cases.append(
            dict(
                name=name,
                role=role,
                frame_indices=frames,
                case_result_ref=ref,
                arrays_ref=case["arrays"],
                endpoint_receipts_ref=case["endpoint_receipts"],
                record=record,
                native_steps=record["native_steps"],
                expected_native_success=[gold[name]["complete_native_success_sequence"][f] for f in frames],
            )
        )
    sources = dict(native["source_sha256"])
    for relative in NEW_SOURCES:
        path = (REPO / relative).resolve()
        require(str(path) not in sources, "new diagnostic source overlaps immutable original423")
        sources[str(path)] = inputs[str(path)] = sha256(path)
    root = (REPO / config["output_dir"]).resolve()
    require(root.is_relative_to(REPO / "outputs") and len(sources) == 431, "diagnostic output/source closure changed")
    return dict(
        schema_version="s2a.native-terminal-predicate-preparation.v1",
        seed=42,
        config=config,
        config_ref=dict(path=str(CONFIG), sha256=CONFIG_SHA),
        registration_ref=dict(path=str(PLAN), sha256=PLAN_SHA),
        output_root=str(root),
        native_prepared=native,
        policy_config=native["policy_config"],
        cases=cases,
        cv2_probe_ref=native["cv2_probe_ref"],
        cv2_runtime_source_refs=native["cv2_runtime_source_refs"],
        runtime_source_refs=native["runtime_source_refs"],
        extra_runtime_source_refs=extra_runtime,
        predicate_runtime_sha256=inventory,
        source_sha256=sources,
        input_sha256=inputs,
        budget=config["budget"],
        **refs,
        **dict.fromkeys(FALSE_FLAGS, False),
    )


def snapshot_mapping(prepared):
    snapshots = dict(prepared["native_prepared"]["snapshot_sha256"])
    directory = Path(prepared["output_root"]) / "preparation/snapshots"
    for index, relative in enumerate(NEW_SOURCES):
        name = f"{index:02d}_" + relative.replace("/", "__")
        snapshots[str(directory / name)] = prepared["source_sha256"][str((REPO / relative).resolve())]
    return snapshots


def prepare():
    cpu_guard()
    metadata = build_metadata()
    root = Path(metadata["output_root"])
    require(not root.exists(), "static diagnostic refuses existing preparation/output root")
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines())
    require(set(NEW_SOURCES) <= tracked, "all eight diagnostic sources must be tracked before preparation")
    snapshots = snapshot_mapping(metadata)
    directory = root / "preparation"
    (directory / "snapshots").mkdir(parents=True)
    for path, digest in snapshots.items():
        destination = Path(path)
        if destination.is_relative_to(directory):
            relative = NEW_SOURCES[int(destination.name[:2])]
            shutil.copyfile(REPO / relative, destination)
        require(sha256(destination) == digest, "frozen diagnostic snapshot differs")
    metadata["snapshot_sha256"] = snapshots
    path = directory / "prepared.json"
    write_once(path, metadata)
    require(read_prepared(path, sha256(path)) == metadata, "published preparation did not revalidate")
    print(json.dumps(dict(prepared=reference(path))))


def read_prepared(path, expected_sha):
    cpu_guard()
    actual = read_ref(dict(path=str(Path(path).resolve()), sha256=expected_sha), {})
    expected = build_metadata()
    expected["snapshot_sha256"] = snapshot_mapping(expected)
    require(actual == expected, "static diagnostic preparation content/source closure changed")
    for target, digest in actual["snapshot_sha256"].items():
        require(sha256(target) == digest, "static diagnostic frozen snapshot changed")
    return actual


def cpu_environment():
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )


def planned_commands(args, prepared):
    root = Path(prepared["output_root"])
    common = ["--prepared", str(args.prepared.resolve()), "--prepared-sha256", args.prepared_sha256, "--seed", "42"]
    worker = [
        str(REPO / prepared["config"]["simulation_python"]),
        "scripts/collect_native_terminal_predicate_diagnostic.py",
        *common,
    ]
    checker = [
        str(REPO / ".venv/bin/python"),
        "scripts/verify_native_terminal_predicate_diagnostic.py",
        *common,
        "--collection",
        str(root / "collection/result.json"),
        "--worker-lifecycle",
        str(root / "orchestration/run/worker_lifecycle.json"),
        "--output",
        str(root / "review/verified.json"),
    ]
    return worker, checker


def execute(args):
    cpu_guard()
    require(args.seed == 42, "static diagnostic seed must be42")
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    commands = planned_commands(args, prepared)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    worker=commands[0], checker=commands[1], cuda_visible_devices="", ld_library_path="", model_calls=0
                )
            )
        )
        return
    root = Path(prepared["output_root"])
    directory = root / "orchestration/run"
    require(
        not any((root / name).exists() for name in ("collection", "review", "orchestration")),
        "static diagnostic refuses existing execution output; no retry",
    )
    directory.mkdir(parents=True)
    (root / "review").mkdir()
    result = dict(
        schema_version="s2a.native-terminal-predicate-owner.v1",
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        owned_commands=[],
        command_attempts=[],
        **dict.fromkeys(FALSE_FLAGS, False),
    )
    try:
        for index, command in enumerate(commands):
            read_prepared(args.prepared, args.prepared_sha256)
            phase = "worker" if index == 0 else "independent"
            attempt = dict(phase=phase, command=command, directory=str(directory / phase))
            result["command_attempts"].append(attempt)
            lifecycle = owned_command(command, cpu_environment(), directory / phase, timeout=900)
            result["owned_commands"].append(lifecycle)
            lifecycle_guard(lifecycle, {})
            if index == 0:
                collection_ref = reference(root / "collection/result.json")
                collection = read_ref(collection_ref, {})
                require(
                    collection.get("complete") is True and collection.get("prepared_ref") == reference(args.prepared),
                    "static worker incomplete or unrelated",
                )
                result["collection_ref"] = collection_ref
                write_once(
                    directory / "worker_lifecycle.json",
                    dict(
                        prepared_ref=reference(args.prepared), collection_ref=collection_ref, worker_lifecycle=lifecycle
                    ),
                )
            else:
                review_ref = reference(root / "review/verified.json")
                review = read_ref(review_ref, {})
                require(
                    review.get("schema_version") == "s2a.native-terminal-predicate-independent-verification.v1"
                    and review.get("passed") is True
                    and review.get("prepared") == reference(args.prepared)
                    and review.get("input_sha256", {}).get(result["collection_ref"]["path"])
                    == result["collection_ref"]["sha256"]
                    and all(review.get(k) is False for k in FALSE_FLAGS if k != "automatic_retry"),
                    "static independent complete evidence missing, failed, or unrelated",
                )
                result["verification_ref"] = review_ref
        read_prepared(args.prepared, args.prepared_sha256)
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        for item in result["command_attempts"]:
            folder = Path(item["directory"])
            item["observed_lifecycle"] = {
                k: reference(folder / name)
                for k, name in (("launch", "launch.json"), ("exit", "exit.json"), ("release", "ownership/release.json"))
                if (folder / name).exists()
            }
        write_once(directory / "result.json", result)
    print(json.dumps(dict(result=reference(directory / "result.json"))))


def interrupted(signum, frame):
    raise InterruptedError(f"static diagnostic owner interrupted by signal{signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    if args.stage == "prepare":
        require(
            not args.dry_run and args.prepared is None and args.prepared_sha256 is None,
            "prepare has no dry-run or inherited prepared argument",
        )
        prepare()
    else:
        require(args.prepared is not None and args.prepared_sha256 is not None, "run requires preparation and SHA")
        execute(args)


if __name__ == "__main__":
    main()
