"""Audit cached native task0 and two isolated continuation tasks, without replay.

Only the SHA-pinned original independent geometry/pixel checker is reused.
Its preparation routine and dynamic Git closure are never called. Original
nonzero exits remain failures; native quality and evidence integrity differ.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256

CONFIG = REPO / "configs/wam_native_training_continuation_20261021.json"
CONFIG_SHA = "75ad3ed9309785bd14be7cde6861ff8a5fb7f9f255fd7a22a7bb2341d8b88e5a"
PLAN = REPO / "plans/v0.0.1/030_environment_continuation_20261021.md"
PLAN_SHA = "53c73c219ede7fe93dc1b1274774a157f138b773fbda64a2ac8bbeaccc7859c7"
OLD_CHECKER = REPO / "scripts/verify_wam_native_training.py"
OLD_CHECKER_SHA = "7b244c8936082d0155ec8d6d034ad94e14ff16f6ecbb24e3cb281d44f6f1831f"
NEW_SOURCES = (
    "plans/v0.0.1/030_environment_continuation_20261021.md",
    "configs/wam_native_training_continuation_20261021.json",
    "scripts/prepare_wam_native_training_continuation.py",
    "tests/test_native_training_continuation_preparation.py",
    "scripts/run_wam_native_training_continuation.py",
    "tests/test_native_training_continuation_runner.py",
    "scripts/collect_wam_native_training_continuation.py",
    "tests/test_native_training_continuation_worker.py",
    "src/s2a/native_environment.py",
    "tests/test_native_environment.py",
    "scripts/verify_wam_native_training_continuation.py",
    "tests/test_native_training_continuation_verification.py",
)
REF_FIELDS = (
    "original_prepared",
    "original_preparation_verification",
    "failed_task",
    "failed_owner",
    "tool_exit",
    "cv2_probe",
)
F64_RUNTIME = ("robots/single_arm.py", "utils/binding_utils.py", "utils/transform_utils.py")
INHERITED = (
    "records",
    "policy_config",
    "tool_contract_ref",
    "tool_source_from_tcp",
    "fresh_config_ref",
    "fresh_fixtures_ref",
    "historical_fixture_refs",
    "selection",
    "forbidden_initial_state_sha256",
    "source_stats",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


class Evidence:
    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        require(path.suffix not in (".hdf5", ".h5"), "continuation must not reopen HDF content")
        digest = sha256(path)
        require(expected is None or digest == expected, f"continuation evidence SHA differs: {path}")
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, f"evidence changed: {path}")
        self.hashes[str(path)] = digest
        return path

    def json(self, path, expected=None):
        return json.loads(self.check(path, expected).read_text())

    def ref(self, ref):
        require(set(ref) == {"path", "sha256"}, "invalid evidence reference")
        return self.json(ref["path"], ref["sha256"])

    def recheck(self):
        for path, digest in self.hashes.copy().items():
            self.check(path, digest)


def load_gold(evidence):
    path = evidence.check(OLD_CHECKER, OLD_CHECKER_SHA)
    spec = importlib.util.spec_from_file_location("frozen_independent_native_continuation_gold", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_stat(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def source_closure(prepared, original, evidence):
    inherited = original["source_sha256"]
    require(len(inherited) == len(original["snapshot_sha256"]) == 409, "original409 closure required")
    expected = dict(inherited)
    for relative in NEW_SOURCES:
        path = str((REPO / relative).resolve())
        require(path not in expected, "new source overlaps original409")
        expected[path] = sha256(path)
    require(
        len(expected) == 421
        and prepared["source_sha256"] == expected
        and prepared["inherited_source_sha256"] == inherited
        and prepared["inherited_snapshot_sha256"] == original["snapshot_sha256"],
        "continuation must preserve literal409 plus exact12 source closure",
    )
    snapshots = dict(original["snapshot_sha256"])
    for index, source in enumerate(sorted(set(expected) - set(inherited))):
        target = Path(prepared["output_root"]) / "preparation/source_snapshot" / f"{index:04d}_{Path(source).name}"
        snapshots[str(target)] = expected[source]
    require(prepared["snapshot_sha256"] == snapshots and len(snapshots) == 421, "421 snapshot/source mapping differs")
    for mapping in (expected, snapshots):
        for path, digest in mapping.items():
            evidence.check(path, digest)


def strict_environment(snapshot):
    require(
        snapshot == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(REPO)),
        "actual continuation cwd/CUDA/LD is not strict CPU environment",
    )


def verify_probe(probe, runtime, evidence):
    strict_environment(probe["before"])
    loader = REPO / "outputs/libero_env311/lib/python3.11/site-packages/cv2/__init__.py"
    expected = [reference(loader.parent / name) for name in ("__init__.py", "config.py", "config-3.py")]
    require(
        runtime == expected
        and probe["cv2_loader_ref"] == expected[0]
        and probe["schema_version"] == "s2a.native-training-import-environment-probe.v1"
        and probe["after_import_cv2"] == dict(probe["before"], LD_LIBRARY_PATH=str(loader.parent / "../../lib64") + ":")
        and probe["reproduces_empty_ld_guard_rejection"] is True
        and probe["original_failed_worker_end_environment_recorded"] is False
        and probe["original_trigger_uniquely_identified"] is False
        and probe["root_original_worker_replayed"] is False
        and all(
            probe[key] == 0
            for key in ("new_environment_initializations", "new_physics_steps", "new_model_calls", "new_hdf_reads")
        ),
        "only the registered cv2 mutation is admitted; original failure cause remains unknown",
    )
    for ref in expected:
        evidence.check(ref["path"], ref["sha256"])


def owned_lifecycle(lifecycle, evidence, *, returncode):
    require(set(lifecycle) == {"launch", "exit", "release"}, "full owned lifecycle required")
    launch, ended, release = (evidence.ref(lifecycle[key]) for key in ("launch", "exit", "release"))
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and launch["automatic_retry"] is False
        and ended == dict(actual_wait=True, automatic_retry=False, returncode=returncode)
        and all(release[key] == launch["child"][key] for key in ("pid", "start_ticks"))
        and release["returncode"] == returncode
        and release["reaped"] is True
        and release["signal_sent"] is False
        and release["only_owned_child_signalled"] is True
        and ((release["original_error"] is None) if returncode == 0 else isinstance(release["original_error"], str)),
        f"owned worker actual wait{returncode}/reap/noSignal identity differs",
    )
    return dict(
        pid=release["pid"], start_ticks=release["start_ticks"], returncode=returncode, reaped=True, signal_sent=False
    )


def verify_original_failure(prepared, original, evidence):
    task, owner, tool = (
        evidence.ref(prepared[key]) for key in ("failed_task_ref", "failed_owner_ref", "tool_exit_ref")
    )
    original_ref, root = prepared["original_prepared_ref"], Path(original["output_root"])
    expected_cases = [
        str(root / "collection/task0" / record["name"] / "result.json") for record in original["records"][:10]
    ]
    require(
        task["schema_version"] == "s2a.native-training-task.v1"
        and task["task_index"] == 0
        and task["prepared_ref"] == original_ref
        and task["seed"] == 42
        and task["complete"] is task["passed"] is False
        and task["error"]["type"] == "ValueError"
        and task["error"]["message"] == "030 source guard requires S2A cwd and empty CUDA/LD"
        and "read_prepared" in task["error"]["traceback"]
        and len(task["cases"]) == task["environment_initializations"] == 10
        and [ref["path"] for ref in task["cases"]] == expected_cases
        and task["physics_steps"] == 1328
        and task["current_canvas_images"] == 50
        and all(
            task[key] == 0
            for key in ("wam_initializations", "denoise_calls", "fits", "fresh_cases", "future_rgb_frames")
        ),
        "original task must remain failed with complete10/1328/50 cached case manifests",
    )
    require(
        owner["schema_version"] == "s2a.native-training-owner.v1"
        and owner["complete"] is owner["passed"] is False
        and owner["prepared_ref"] == original_ref
        and owner["source_peer_ref"] == prepared["original_preparation_verification_ref"]
        and owner["automatic_retry"] is False
        and owner["owned_commands"] == []
        and len(owner["command_attempts"]) == 1
        and owner["command_attempts"][0]["task_index"] == 0
        and owner["command_attempts"][0]["phase"] == "collection"
        and owner["error"]["type"] == "ChildProcessError"
        and "failed with1" in owner["error"]["message"],
        "original owner must remain failed with exactly one task0 attempt",
    )
    lifecycle = owner["command_attempts"][0]["observed_lifecycle"]
    require(prepared["original_lifecycle_refs"] == lifecycle, "cached original lifecycle binding differs")
    child = owned_lifecycle(lifecycle, evidence, returncode=1)
    require(
        tool["schema_version"] == "s2a.native-training-owner-tool-exit-observation.v1"
        and tool["original_owner_result_ref"] == prepared["failed_owner_ref"]
        and tool["original_task_result_ref"] == prepared["failed_task_ref"]
        and tool["prepared_ref"] == original_ref
        and tool["task_release_ref"] == lifecycle["release"]
        and tool["actual_tool_terminal_exit_code"] == tool["actual_task_wait_returncode"] == 1
        and tool["actual_task_reaped"] is tool["owner_birth_absent"] is tool["task_birth_absent"] is True
        and tool["actual_task_signal_sent"] is tool["root_manual_popen_wait_claimed"] is False
        and tool["complete_case_refs"] == task["cases"] == prepared["cached_case_refs"]
        and tool["actual_physics_steps"] == 1328
        and tool["actual_current_canvases"] == 50
        and tool["complete_source_guard_passed"]
        is tool["full30_collection_certified"]
        is tool["independent_case_quality_verified"]
        is False
        and tool["prior409_source_bytes_unchanged"] is True
        and tool["new_model_calls"] == tool["fresh_cases"] == 0
        and tool["automatic_retry"] is False,
        "original nonzero outer tool terminal and owned child release were not preserved",
    )
    require(not (root / "collection/result.json").exists(), "original public success unexpectedly exists")
    for index in (1, 2):
        require(
            not (root / "collection" / f"task{index}").exists()
            and not (root / "orchestration/collection" / f"task{index}").exists(),
            "original remaining task was already started",
        )
    for ref in task["cases"]:
        evidence.check(ref["path"], ref["sha256"])
    return task, child


def verify_preparation(path, expected_sha, evidence):
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    prepared = evidence.json(path, expected_sha)
    root = (REPO / config["output_dir"]).resolve()
    require(
        prepared["schema_version"] == "s2a.native-training-continuation-preparation.v1"
        and prepared["seed"] == config["seed"] == 42
        and prepared["config"] == config
        and prepared["config_ref"] == dict(path=str(CONFIG), sha256=CONFIG_SHA)
        and prepared["registration_ref"] == dict(path=str(PLAN), sha256=PLAN_SHA)
        and root == REPO / "outputs/wam_stability_20261021/synchronous_native_training_continuation"
        and prepared["output_root"] == str(root)
        and Path(path).resolve() == root / "preparation/prepared.json",
        "continuation preparation/config/registration/root binding differs",
    )
    payloads = {}
    for name in REF_FIELDS:
        expected = dict(path=str((REPO / config[name]).resolve()), sha256=config[name + "_sha256"])
        require(prepared[name + "_ref"] == expected, f"registered {name} identity differs")
        payloads[name] = evidence.ref(expected)
    original, peer = payloads["original_prepared"], payloads["original_preparation_verification"]
    require(
        original["schema_version"] == "s2a.native-training-preparation.v1"
        and len(original["input_sha256"]) == 49
        and original["metadata_admission"] is True
        and original["native_success_verified"]
        is original["model_training_authorized"]
        is original["fresh_performance_authorized"]
        is False
        and peer["schema_version"] == "s2a.native-training-independent-verification.v1"
        and peer["phase"] == "prepare"
        and peer["passed"] is peer["preparation_verified"] is True
        and peer["prepared"] == prepared["original_prepared_ref"]
        and all(peer["input_sha256"].get(key) == value for key, value in original["input_sha256"].items()),
        "original complete metadata admission and non-authorization required",
    )
    require(
        prepared["original_prepared"] == original
        and prepared["original_config"] == original["config"]
        and prepared["original_output_root"] == original["output_root"]
        and all(prepared[key] == original[key] for key in INHERITED)
        and len(original["records"]) == 30
        and [r["task_index"] for r in original["records"]] == [i for i in range(3) for _ in range(10)]
        and len({r["name"] for r in original["records"]}) == 30
        and prepared["cached_records"] == original["records"][:10]
        and prepared["remaining_records"] == original["records"][10:]
        and prepared["remaining_tasks"] == [1, 2]
        and sum(r["native_steps"] + 40 for r in prepared["cached_records"]) == 1328
        and sum(r["native_steps"] + 40 for r in prepared["remaining_records"]) == 3652,
        "continuation cannot change/reselect the fixed30 records",
    )
    source_closure(prepared, original, evidence)
    require(
        prepared["original_checker_ref"] == dict(path=str(OLD_CHECKER), sha256=OLD_CHECKER_SHA),
        "unfrozen geometry gold",
    )
    require(
        original["source_sha256"].get(str(OLD_CHECKER)) == OLD_CHECKER_SHA, "original independent gold identity differs"
    )
    runtime = [
        reference(REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite" / name) for name in F64_RUNTIME
    ]
    require(original["runtime_source_refs"] == runtime, "original F64 physical observation source chain differs")
    verify_probe(payloads["cv2_probe"], prepared["cv2_runtime_source_refs"], evidence)
    require(
        prepared["runtime_source_refs"] == runtime + prepared["cv2_runtime_source_refs"],
        "actual runtime source roster differs",
    )
    task, child = verify_original_failure(prepared, original, evidence)
    expected_inputs = dict(original["input_sha256"])
    expected_inputs[str(CONFIG)] = CONFIG_SHA
    for ref in [
        *(prepared[name + "_ref"] for name in REF_FIELDS),
        *prepared["original_lifecycle_refs"].values(),
        *task["cases"],
        *prepared["cv2_runtime_source_refs"],
    ]:
        require(
            ref["path"] not in expected_inputs or expected_inputs[ref["path"]] == ref["sha256"],
            "inherited SHA rewritten",
        )
        expected_inputs[ref["path"]] = ref["sha256"]
    require(prepared["input_sha256"] == expected_inputs, "continuation input map was expanded or changed")
    hdf = original["source_stats"]
    require(len(hdf) == 3 and all(source_stat(p) == stat for p, stat in hdf.items()), "inherited HDF stat changed")
    stat_only = [dict(path=p, sha256=expected_inputs[p], stat=hdf[p]) for p in sorted(hdf)]
    require(prepared["inherited_hdf5_refs_stat_only"] == stat_only, "HDF stat-only identity disclosure differs")
    expert_paths = {r["expert_path"] for r in original["records"]}
    for source, digest in expected_inputs.items():
        if source not in hdf:
            require(
                Path(source).suffix not in (".npz", ".png", ".h5", ".hdf5")
                and (Path(source).suffix != ".npy" or source in expert_paths),
                "unsafe inherited observation/fresh action input",
            )
            evidence.check(source, digest)
    actual = dict(expected_inputs)
    for source in hdf:
        del actual[source]
    require(prepared["actual_input_sha256"] == actual, "actual safe-input identity map differs")
    require(
        prepared["budget"]
        == dict(
            cached_cases=10,
            cached_physics_steps=1328,
            cached_current_canvases=50,
            environment_initializations=20,
            physics_steps=3652,
            current_canvas_images=100,
            whole_collection_cases=30,
            whole_collection_physics_steps=4980,
            whole_collection_current_canvases=150,
            wam_calls=0,
            fits=0,
            fresh_cases=0,
        )
        and prepared["metadata_admission"] is True
        and all(
            prepared[key] is False
            for key in (
                "original_task_passed",
                "original_owner_passed",
                "original_end_environment_known",
                "cached_quality_verified",
                "eligible_for_model_training_registration",
                "model_training_authorized",
                "fresh_performance_authorized",
            )
        )
        and prepared["new_physics_steps"] == 0,
        "continuation preparation promoted cached/native quality or exceeded finite budget",
    )
    return prepared, original, task, child


def audit_cases(refs, records, original, directory, evidence, gold):
    require(len(refs) == len(records) == 10, "each fixed task must contain all10 case refs")
    rows = []
    for ref, record in zip(refs, records, strict=True):
        require(
            ref["path"] == str(Path(directory) / record["name"] / "result.json"), "case escaped exact source/task path"
        )
        native_path = evidence.check(record["expert_path"], record["expert_sha256"])
        native = np.load(native_path, allow_pickle=False)
        require(
            native.shape == (record["native_steps"], 7)
            and str(native.dtype) == record["expert_dtype"]
            and np.isfinite(native).all(),
            "original selected expert dtype/shape/nonfinite differs",
        )
        rows.append(gold.case_contract(evidence.ref(ref), record, original, native, evidence))
    return rows


def verify_prefix(prepared, original, task, child, evidence, gold):
    rows = audit_cases(
        task["cases"],
        prepared["cached_records"],
        original,
        Path(original["output_root"]) / "collection/task0",
        evidence,
        gold,
    )
    require(sum(row["physics_steps"] for row in rows) == 1328, "cached physics budget differs")
    return dict(
        prefix_verified=True,
        evidence_integrity_passed=True,
        cached_case_refs=task["cases"],
        case_verification=rows,
        original_task_passed=False,
        original_owner_passed=False,
        original_worker_returncode=1,
        original_owned_worker=child,
        cached_physics_steps=1328,
        cached_current_canvas_images=50,
        cached_native_quality=dict(
            total_cases=10,
            stable_native_cases=sum(row["stable_native_success"] for row in rows),
            failed_native_cases=[row["name"] for row in rows if not row["stable_native_success"]],
        ),
        interpretation="These10 saved cases were audited without replay. The original task/owner still failed; its ending environment and unique trigger are unknown. No complete30 or model-training authorization follows.",
    )


def verify_readiness(ref, prepared, launch, evidence):
    actual = evidence.ref(ref)
    probe = evidence.ref(prepared["cv2_probe_ref"])
    sources = {item["path"]: item["sha256"] for item in prepared["cv2_runtime_source_refs"]}
    require(
        actual["schema_version"] == "s2a.native-training-environment-readiness.v1"
        and actual["complete"] is actual["passed"] is actual["restore_performed"] is True
        and "error" not in actual
        and actual["continuation_prepared_ref"]
        == reference(Path(prepared["output_root"]) / "preparation/prepared.json")
        and actual["cv2_probe_ref"] == prepared["cv2_probe_ref"]
        and actual["cv2_runtime_source_refs"] == prepared["cv2_runtime_source_refs"]
        and actual["before"] == probe["before"]
        and actual["after_import_cv2"] == probe["after_import_cv2"]
        and actual["after_restore"] == probe["before"]
        and actual["source_before_sha256"] == actual["source_after_sha256"] == sources
        and actual["actual_cv2_loader"] == probe["cv2_loader_ref"]["path"]
        and actual["actual_cv2_version"] == probe["cv2_version"]
        and actual["actual_cv2_loader_ref"] == probe["cv2_loader_ref"]
        and actual["bytecode_before"] == actual["bytecode_after_import"] == actual["bytecode_after_restore"] == "1"
        and actual["pid"] == launch["child"]["pid"]
        and actual["argv"] == launch["command"][1:]
        and all(
            actual[key] == 0
            for key in ("new_environment_initializations", "new_physics_steps", "new_model_calls", "new_hdf_reads")
        ),
        "actual cv2 admission/known mutation/strict restore/source/process ordering differs",
    )
    strict_environment(actual["before"])
    strict_environment(actual["after_restore"])
    return actual


def case_environment(ref, name, phase, prepared_ref, child_pid, evidence):
    actual = evidence.ref(ref)
    require(
        actual["schema_version"] == "s2a.native-training-case-environment.v1"
        and actual["name"] == name
        and actual["phase"] == phase
        and actual["continuation_prepared_ref"] == prepared_ref
        and actual["passed"] is True
        and "error" not in actual
        and actual["bytecode"] == "1"
        and actual["pid"] == child_pid
        and isinstance(actual["argv"], list),
        "actual per-case strict environment/process evidence differs",
    )
    strict_environment(actual["actual_environment"])


def verified_prefix(path, prepared, original_task, evidence):
    prefix = evidence.json(path)
    prepared_ref = reference(Path(prepared["output_root"]) / "preparation/prepared.json")
    require(
        Path(path).resolve() == Path(prepared["output_root"]) / "review/prefix_verified.json"
        and prefix["schema_version"] == "s2a.native-training-continuation-independent-verification.v1"
        and prefix["phase"] == "prefix"
        and prefix["passed"] is prefix["prefix_verified"] is prefix["evidence_integrity_passed"] is True
        and "error" not in prefix
        and prefix["prepared"] == prepared_ref
        and prefix["input_sha256"].get(prepared_ref["path"]) == prepared_ref["sha256"]
        and prefix["cached_case_refs"] == original_task["cases"]
        and prefix["cached_physics_steps"] == 1328
        and prefix["cached_current_canvas_images"] == 50
        and prefix["original_worker_returncode"] == 1
        and prefix["original_task_passed"] is prefix["original_owner_passed"] is False
        and all(
            prefix[key] is False
            for key in (
                "eligible_for_model_training_registration",
                "model_training_authorized",
                "fresh_performance_authorized",
            )
        ),
        "final collection requires this preparation's complete prefix audit, not an original success",
    )
    # Reuse the original scores, but not a stale promise about their inputs.
    # Hashing a saved NPZ/PNG does not decode it or repeat case scoring.
    for consumed_path, digest in prefix["input_sha256"].items():
        evidence.check(consumed_path, digest)
    rows = prefix["case_verification"]
    require(
        len(rows) == 10
        and [row["name"] for row in rows] == [r["name"] for r in prepared["cached_records"]]
        and all(row["task_index"] == 0 and type(row["stable_native_success"]) is bool for row in rows)
        and sum(row["physics_steps"] for row in rows) == 1328,
        "cached prefix summary lost fixed10 denominator/origin",
    )
    return rows


def verify_collection(path, prefix_path, prepared, original, original_task, evidence, gold):
    cached = verified_prefix(prefix_path, prepared, original_task, evidence)
    actual = evidence.json(path)
    prepared_ref = reference(Path(prepared["output_root"]) / "preparation/prepared.json")
    prefix_ref = reference(prefix_path)
    require(
        actual["schema_version"] == "s2a.native-training-continuation-collection.v1"
        and actual["complete"] is actual["passed"] is True
        and "error" not in actual
        and actual["prepared_ref"] == prepared_ref
        and actual["original_prepared_ref"] == prepared["original_prepared_ref"]
        and actual["prefix_verification_ref"] == prefix_ref
        and actual["cached_task_ref"] == prepared["failed_task_ref"]
        and actual["environment_initializations"] == 30
        and actual["physics_steps"] == 4980
        and actual["current_canvas_images"] == 150
        and actual["new_environment_initializations"] == 20
        and actual["new_collection_physics_steps"] == 3652
        and actual["new_current_canvas_images"] == 100
        and actual["cached_cases"] == 10
        and actual["cached_physics_steps"] == 1328
        and actual["cached_current_canvas_images"] == 50
        and actual["automatic_retry"]
        is actual["model_training_authorized"]
        is actual["fresh_performance_authorized"]
        is False
        and all(actual[key] == 0 for key in ("wam_initializations", "denoise_calls", "fits", "fresh_cases"))
        and len(actual["task_results"]) == 2
        and [item["task_index"] for item in actual["task_results"]] == [1, 2],
        "continuation public collection must preserve cached failure and exact2 new-task/30 whole budget",
    )
    # Both new producers must be actually waited/reaped before any new labels
    # or native success are read. The old task has a separate exit1 origin.
    children = [owned_lifecycle(item["lifecycle"], evidence, returncode=0) for item in actual["task_results"]]
    rows = list(cached)
    for item, child in zip(actual["task_results"], children, strict=True):
        index = item["task_index"]
        directory = Path(prepared["output_root"]) / "collection" / f"task{index}"
        require(item["result"]["path"] == str(directory / "result.json"), "new task result escaped exact root")
        task = evidence.ref(item["result"])
        require(
            task["schema_version"] == "s2a.native-training-continuation-task.v1"
            and task["complete"] is task["passed"] is True
            and "error" not in task
            and task["task_index"] == index
            and task["seed"] == 42
            and task["prepared_ref"] == prepared["original_prepared_ref"]
            and task["continuation_prepared_ref"] == prepared_ref
            and task["actual_environment"]
            == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
            and task["environment_initializations"] == 10
            and task["current_canvas_images"] == 50
            and task["automatic_retry"] is False
            and len(task["case_environment"]) == 10
            and all(
                task[key] == 0
                for key in (
                    "cached_cases_recollected",
                    "wam_initializations",
                    "denoise_calls",
                    "fits",
                    "fresh_cases",
                    "future_rgb_frames",
                    "source_hdf5_reads",
                )
            ),
            "new task origin/budget/strict environment differs",
        )
        launch = evidence.ref(item["lifecycle"]["launch"])
        require(
            task["environment_readiness_ref"]["path"] == str(directory / "environment_readiness.json"),
            "readiness root differs",
        )
        readiness = verify_readiness(task["environment_readiness_ref"], prepared, launch, evidence)
        started = evidence.json(directory / "worker_started.json")
        require(
            started["pid"] == child["pid"]
            and started["argv"] == readiness["argv"]
            and started["environment_readiness_ref"] == task["environment_readiness_ref"]
            and started["prepared_ref"] == prepared["original_prepared_ref"]
            and started["continuation_prepared_ref"] == prepared_ref
            and started["task_index"] == index
            and started["cuda"] == started["ld"] == ""
            and started["actual_environment"] == task["actual_environment"],
            "worker startup was not bound to successful pre-simulator environment restore",
        )
        records = original["records"][10 * index : 10 * (index + 1)]
        for record, environment in zip(records, task["case_environment"], strict=True):
            require(environment["name"] == record["name"], "per-case environment order differs")
            for phase, field in (("before_case", "before_ref"), ("after_case", "after_ref")):
                suffix = "before" if field == "before_ref" else "after"
                require(
                    environment[field]["path"] == str(directory / "environment" / f"{record['name']}_{suffix}.json"),
                    "case environment escaped exact path",
                )
                case_environment(environment[field], record["name"], phase, prepared_ref, child["pid"], evidence)
        members = audit_cases(task["cases"], records, original, directory, evidence, gold)
        require(
            task["physics_steps"] == sum(row["physics_steps"] for row in members)
            and task["native_quality_passed"] is all(row["stable_native_success"] for row in members),
            "new task native quality/physics aggregate differs",
        )
        rows.extend(members)
    require(sum(row["physics_steps"] for row in rows) == 4980, "complete30 physical denominator differs")
    quality = gold.quality_summary(rows)
    return dict(
        collection_verified=True,
        evidence_integrity_passed=True,
        prefix_verification_ref=prefix_ref,
        cached_case_artifacts_rescored=False,
        cached_summary_reused_cases=10,
        newly_audited_cases=20,
        original_task_passed=False,
        original_owner_passed=False,
        original_worker_returncode=1,
        case_verification=rows,
        native_quality=quality,
        eligible_for_model_training_registration=quality["eligible_for_model_training_registration"],
        actual_owned_workers=children,
        physics_steps=4980,
        new_collection_physics_steps=3652,
        environment_initializations=30,
        current_canvas_images=150,
        future_rgb_frames=0,
        interpretation="Full30 native quality uses a separately audited exit1 cache and20 normally released new cases. Quality failure keeps all30. Registration eligibility is not model training authorization, WAM task success, or fresh evaluation.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefix", "collection"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--prefix-verification", type=Path)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()
    strict_environment(
        dict(
            CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
            LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH"),
            cwd=str(Path.cwd().resolve()),
        )
    )
    require(
        (args.phase == "collection") == (args.collection is not None) == (args.prefix_verification is not None),
        "explicit phase/evidence arguments differ",
    )
    output = args.output.resolve()
    require(
        output.is_relative_to(REPO / "outputs") and not output.exists(),
        "new audit output under outputs required; refuses overwrite",
    )
    evidence = Evidence()
    receipt = dict(
        schema_version="s2a.native-training-continuation-independent-verification.v1",
        phase=args.phase,
        passed=False,
        eligible_for_model_training_registration=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        new_physics_steps=0,
        new_wam_calls=0,
        new_fits=0,
        new_parameter_reconstructions=0,
        hdf_datasets_read=0,
        fresh_action_datasets_read=0,
    )
    error = None
    try:
        prepared, original, task, child = verify_preparation(args.prepared, args.prepared_sha256, evidence)
        gold = load_gold(evidence)
        receipt["prepared"] = reference(args.prepared)
        if args.phase == "prefix":
            receipt.update(verify_prefix(prepared, original, task, child, evidence, gold))
        else:
            receipt.update(
                verify_collection(args.collection, args.prefix_verification, prepared, original, task, evidence, gold)
            )
        evidence.recheck()
        receipt["passed"] = True
    except Exception as caught:
        error = caught
        receipt["error"] = dict(type=type(caught).__name__, message=str(caught))
        receipt["eligible_for_model_training_registration"] = False
        receipt["evidence_integrity_passed"] = False
        receipt[args.phase + "_verified"] = False
    receipt["input_sha256"] = evidence.hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=receipt["passed"], output=reference(output))))
    if error is not None:
        raise SystemExit(f"030 continuation independent audit rejected: {error}")


if __name__ == "__main__":
    main()
