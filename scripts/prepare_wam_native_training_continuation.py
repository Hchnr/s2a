"""Prepare the isolated030 continuation without HDF reads or replaying task0.

The original409 source set is literal evidence, not a query of today's tracked
files. Existing HDF identities remain stat-only inherited references.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/wam_native_training_continuation_20261021.json"
CONFIG_SHA = "75ad3ed9309785bd14be7cde6861ff8a5fb7f9f255fd7a22a7bb2341d8b88e5a"
PLAN = REPO / "plans/v0.0.1/030_environment_continuation_20261021.md"
PLAN_SHA = "53c73c219ede7fe93dc1b1274774a157f138b773fbda64a2ac8bbeaccc7859c7"
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
CV2_RUNTIME_SOURCE_PATHS = tuple(
    "outputs/libero_env311/lib/python3.11/site-packages/cv2/" + name
    for name in ("__init__.py", "config.py", "config-3.py")
)
F64_RUNTIME_SOURCE_PATHS = (
    "outputs/libero_env311/lib/python3.11/site-packages/robosuite/robots/single_arm.py",
    "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/binding_utils.py",
    "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/transform_utils.py",
)
REF_FIELDS = (
    ("original_prepared", "original_prepared_ref"),
    ("original_preparation_verification", "original_preparation_verification_ref"),
    ("failed_task", "failed_task_ref"),
    ("failed_owner", "failed_owner_ref"),
    ("tool_exit", "tool_exit_ref"),
    ("cv2_probe", "cv2_probe_ref"),
)
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


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def source_stat(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def checked_json(path, expected_sha, actual_inputs):
    path = Path(path).resolve()
    require(sha256(path) == expected_sha, f"continuation JSON source changed: {path}")
    actual_inputs[str(path)] = expected_sha
    return json.loads(path.read_text())


def cpu_guard():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "continuation requires S2A cwd and explicitly empty CUDA/LD",
    )


def assert_config(config):
    expected = dict(
        seed=42,
        remaining_tasks=[1, 2],
        cached_cases=10,
        new_cases=20,
        new_physics_steps=3652,
        new_current_canvases=100,
        whole_collection_cases=30,
        whole_collection_physics_steps=4980,
        whole_collection_canvases=150,
        wam_calls=0,
        fits=0,
        fresh_cases=0,
        automatic_retry=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
    )
    require(
        all(config[key] == value for key, value in expected.items()),
        "continuation registered budget or task split changed",
    )
    require(
        config["output_dir"] == "outputs/wam_stability_20261021/synchronous_native_training_continuation",
        "continuation output mapping differs",
    )


def original_guard(original, original_ref, peer, actual_inputs):
    """No dynamic source enumeration and no inherited HDF byte reads."""
    require(
        original["schema_version"] == "s2a.native-training-preparation.v1"
        and original["seed"] == 42
        and len(original["source_sha256"]) == len(original["snapshot_sha256"]) == 409
        and len(original["input_sha256"]) == 49
        and len(original["records"]) == 30
        and original["metadata_admission"] is True
        and original["native_success_verified"] is False
        and original["model_training_authorized"] is False
        and original["fresh_performance_authorized"] is False,
        "original409/49 preparation or original non-authorization changed",
    )
    require(
        peer["schema_version"] == "s2a.native-training-independent-verification.v1"
        and peer["phase"] == "prepare"
        and peer["passed"] is True
        and peer["preparation_verified"] is True
        and peer["prepared"] == original_ref
        and peer["input_sha256"].get(original_ref["path"]) == original_ref["sha256"]
        and peer["model_training_authorized"] is False
        and peer["fresh_performance_authorized"] is False,
        "original complete metadata admission required",
    )
    for field in ("source_sha256", "snapshot_sha256"):
        for path, digest in original[field].items():
            require(sha256(path) == digest, f"original409 {field} changed: {path}")
    experts = {str(Path(record["expert_path"]).resolve()) for record in original["records"]}
    hdf_sources = {str(Path(record["source_hdf5"]).resolve()) for record in original["records"]}
    require(
        len(experts) == 30
        and len(hdf_sources) == len(original["source_stats"]) == 3
        and set(original["source_stats"]) == hdf_sources,
        "original source/expert roster differs",
    )
    stat_only = []
    for path, digest in original["input_sha256"].items():
        suffix = Path(path).suffix
        require(
            suffix not in (".png", ".npz") and (suffix != ".npy" or path in experts),
            "inherited observation/fresh expert read forbidden",
        )
        if path in hdf_sources:
            require(source_stat(path) == original["source_stats"][path], "original native HDF stat changed")
            stat_only.append(dict(path=path, sha256=digest, stat=original["source_stats"][path]))
        else:
            require(suffix not in (".hdf5", ".h5") and sha256(path) == digest, f"original safe input changed: {path}")
            actual_inputs[path] = digest
    require(len(stat_only) == 3, "all original HDF identities must remain stat-only")
    expected_runtime = {str((REPO / path).resolve()) for path in F64_RUNTIME_SOURCE_PATHS}
    require(
        {ref["path"] for ref in original["runtime_source_refs"]} == expected_runtime,
        "original F64 observation chain changed",
    )
    for ref in original["runtime_source_refs"]:
        require(
            original["input_sha256"].get(ref["path"]) == ref["sha256"] and sha256(ref["path"]) == ref["sha256"],
            "original runtime source identity differs",
        )
    return stat_only


def failure_admission(original, refs, task, owner, observed, actual_inputs):
    root = Path(original["output_root"])
    original_ref = refs["original_prepared_ref"]
    records = [record for record in original["records"] if record["task_index"] == 0]
    expected_cases = [str(root / "collection" / "task0" / record["name"] / "result.json") for record in records]
    require(
        task["schema_version"] == "s2a.native-training-task.v1"
        and task["task_index"] == 0
        and task["seed"] == 42
        and task["prepared_ref"] == original_ref
        and task["complete"] is False
        and task["passed"] is False
        and task["error"]["type"] == "ValueError"
        and task["error"]["message"] == "030 source guard requires S2A cwd and empty CUDA/LD"
        and len(task["cases"]) == 10
        and [ref["path"] for ref in task["cases"]] == expected_cases
        and task["environment_initializations"] == 10
        and task["physics_steps"] == 1328
        and task["current_canvas_images"] == 50,
        "only the original complete cached10 with terminal guard False may continue",
    )
    require(
        owner["schema_version"] == "s2a.native-training-owner.v1"
        and owner["complete"] is False
        and owner["passed"] is False
        and owner["prepared_ref"] == original_ref
        and owner["source_peer_ref"] == refs["original_preparation_verification_ref"]
        and owner["automatic_retry"] is False
        and owner["owned_commands"] == []
        and len(owner["command_attempts"]) == 1
        and owner["command_attempts"][0]["task_index"] == 0,
        "original failed owner/unique task0 attempt differs",
    )
    require(
        observed["original_owner_result_ref"] == refs["failed_owner_ref"]
        and observed["original_task_result_ref"] == refs["failed_task_ref"]
        and observed["prepared_ref"] == original_ref
        and observed["actual_tool_terminal_exit_code"] == 1
        and observed["root_manual_popen_wait_claimed"] is False
        and observed["actual_task_wait_returncode"] == 1
        and observed["actual_task_reaped"] is True
        and observed["actual_task_signal_sent"] is False
        and observed["owner_birth_absent"] is True
        and observed["task_birth_absent"] is True
        and observed["complete_case_refs"] == task["cases"]
        and observed["actual_physics_steps"] == 1328
        and observed["actual_current_canvases"] == 50
        and observed["complete_source_guard_passed"] is False
        and observed["full30_collection_certified"] is False
        and observed["independent_case_quality_verified"] is False
        and observed["prior409_source_bytes_unchanged"] is True,
        "original actual nonzero exit/reap observation differs",
    )
    lifecycle = owner["command_attempts"][0]["observed_lifecycle"]
    payload = {}
    for key, ref in lifecycle.items():
        expected = (
            root
            / "orchestration"
            / "collection"
            / "task0"
            / ("ownership/release.json" if key == "release" else f"{key}.json")
        )
        require(
            key in ("launch", "exit", "release") and ref["path"] == str(expected),
            "original lifecycle path escaped task0",
        )
        payload[key] = checked_json(ref["path"], ref["sha256"], actual_inputs)
    require(set(payload) == {"launch", "exit", "release"}, "original full lifecycle required")
    launch, exit_record, release = (payload[key] for key in ("launch", "exit", "release"))
    require(
        observed["task_release_ref"] == lifecycle["release"]
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and exit_record["actual_wait"] is True
        and exit_record["returncode"] == 1
        and release["returncode"] == 1
        and release["reaped"] is True
        and release["signal_sent"] is False
        and all(release[key] == launch["child"][key] for key in ("pid", "start_ticks")),
        "original worker wait1/reap/noSignal lifecycle required",
    )
    for ref in task["cases"]:
        require(sha256(ref["path"]) == ref["sha256"], "cached case manifest changed; quality remains unverified")
        actual_inputs[ref["path"]] = ref["sha256"]
    require(
        not (root / "collection" / "result.json").exists(), "original successful public collection unexpectedly exists"
    )
    for task_index in (1, 2):
        require(
            not (root / "collection" / f"task{task_index}").exists()
            and not (root / "orchestration" / "collection" / f"task{task_index}").exists(),
            "original remaining task was already started",
        )
    return lifecycle


def probe_admission(probe, actual_inputs):
    before = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(REPO))
    after = dict(
        before, LD_LIBRARY_PATH=str(REPO / "outputs/libero_env311/lib/python3.11/site-packages/cv2/../../lib64") + ":"
    )
    refs = [reference(REPO / path) for path in CV2_RUNTIME_SOURCE_PATHS]
    require(
        probe["schema_version"] == "s2a.native-training-import-environment-probe.v1"
        and probe["before"] == before
        and probe["after_import_cv2"] == after
        and probe["cv2_loader_ref"] == refs[0]
        and probe["reproduces_empty_ld_guard_rejection"] is True
        and probe["original_failed_worker_end_environment_recorded"] is False
        and probe["original_trigger_uniquely_identified"] is False
        and all(
            probe[key] == 0
            for key in ("new_environment_initializations", "new_physics_steps", "new_model_calls", "new_hdf_reads")
        )
        and probe["root_original_worker_replayed"] is False,
        "exact cv2-only import probe required; original cause remains unknown",
    )
    for ref in refs:
        actual_inputs[ref["path"]] = ref["sha256"]
    return refs


def source_closure(original):
    sources = dict(original["source_sha256"])
    require(len(sources) == 409 and len(NEW_SOURCES) == 12, "literal old409 plus registered12 sources required")
    tracked = tracked_new_sources()
    require(tracked == set(NEW_SOURCES), "all twelve isolated sources must be tracked before preparation")
    for relative in NEW_SOURCES:
        path = (REPO / relative).resolve()
        require(
            path.is_relative_to(REPO) and str(path) not in sources and path.is_file(),
            f"isolated continuation source missing or overlaps old409: {path}",
        )
        sources[str(path)] = sha256(path)
    require(len(sources) == 421, "continuation requires exact409+12 closure")
    return dict(sorted(sources.items()))


def tracked_new_sources():
    """Check only the registered twelve names; never enumerate a new old closure."""
    return set(subprocess.check_output(["git", "ls-files", "--", *NEW_SOURCES], cwd=REPO, text=True).splitlines())


def snapshot_closure(original, root, sources):
    snapshots = dict(original["snapshot_sha256"])
    new_paths = sorted(set(sources) - set(original["source_sha256"]))
    snapshots.update(
        {
            str(root / "preparation" / "source_snapshot" / f"{index:04d}_{Path(source).name}"): sources[source]
            for index, source in enumerate(new_paths)
        }
    )
    return dict(sorted(snapshots.items()))


def evidence(config):
    actual_inputs, refs, payload = {}, {}, {}
    for name, field in REF_FIELDS:
        ref = dict(path=str((REPO / config[name]).resolve()), sha256=config[name + "_sha256"])
        refs[field] = ref
        payload[name] = checked_json(ref["path"], ref["sha256"], actual_inputs)
    original = payload["original_prepared"]
    stat_only = original_guard(
        original, refs["original_prepared_ref"], payload["original_preparation_verification"], actual_inputs
    )
    lifecycle = failure_admission(
        original, refs, payload["failed_task"], payload["failed_owner"], payload["tool_exit"], actual_inputs
    )
    cv2_refs = probe_admission(payload["cv2_probe"], actual_inputs)
    return original, refs, actual_inputs, stat_only, lifecycle, cv2_refs, payload["failed_task"]["cases"]


def build(config, original, refs, actual_inputs, stat_only, lifecycle, cv2_refs, cached_case_refs, sources, snapshots):
    records = original["records"]
    remaining = [record for record in records if record["task_index"] in (1, 2)]
    cached = [record for record in records if record["task_index"] == 0]
    require(
        len(cached) == 10
        and len(remaining) == 20
        and [r["task_index"] for r in records] == [task for task in range(3) for _ in range(10)],
        "original fixed30 record order differs",
    )
    require(
        sum(r["native_steps"] + 40 for r in cached) == 1328
        and sum(r["native_steps"] + 40 for r in remaining) == 3652
        and all(len(r["anchor_frames"]) == 5 for r in records),
        "literal original remaining3652/100 budget differs",
    )
    actual_inputs[str(CONFIG)] = CONFIG_SHA
    inputs = dict(original["input_sha256"])
    for path, digest in actual_inputs.items():
        require(path not in inputs or inputs[path] == digest, "inherited input SHA was rewritten")
        inputs[path] = digest
    checker_path = str(REPO / "scripts/verify_wam_native_training.py")
    require(checker_path in original["source_sha256"], "original independent case checker absent")
    return dict(
        schema_version="s2a.native-training-continuation-preparation.v1",
        seed=42,
        config=config,
        config_ref=dict(path=str(CONFIG), sha256=CONFIG_SHA),
        registration_ref=dict(path=str(PLAN), sha256=PLAN_SHA),
        output_root=str((REPO / config["output_dir"]).resolve()),
        **refs,
        original_prepared=copy.deepcopy(original),
        original_config=original["config"],
        original_output_root=original["output_root"],
        **{key: copy.deepcopy(original[key]) for key in INHERITED},
        cached_records=copy.deepcopy(cached),
        remaining_records=copy.deepcopy(remaining),
        remaining_tasks=[1, 2],
        cached_case_refs=cached_case_refs,
        original_lifecycle_refs=lifecycle,
        original_checker_ref=dict(path=checker_path, sha256=original["source_sha256"][checker_path]),
        source_sha256=sources,
        snapshot_sha256=snapshots,
        inherited_source_sha256=original["source_sha256"],
        inherited_snapshot_sha256=original["snapshot_sha256"],
        input_sha256=inputs,
        actual_input_sha256=actual_inputs,
        inherited_hdf5_refs_stat_only=stat_only,
        cv2_runtime_source_refs=cv2_refs,
        runtime_source_refs=original["runtime_source_refs"] + cv2_refs,
        budget=dict(
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
        ),
        original_task_passed=False,
        original_owner_passed=False,
        original_end_environment_known=False,
        cached_quality_verified=False,
        metadata_admission=True,
        eligible_for_model_training_registration=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        new_physics_steps=0,
    )


def prepare(config_path=CONFIG, *, seed=42):
    cpu_guard()
    require(Path(config_path).resolve() == CONFIG and seed == 42, "registered continuation config/seed required")
    config = checked_json(CONFIG, CONFIG_SHA, {})
    assert_config(config)
    require(sha256(PLAN) == PLAN_SHA, "continuation plan changed")
    root = (REPO / config["output_dir"]).resolve()
    require(
        root.is_relative_to(REPO / "outputs") and not root.exists(),
        "continuation refuses existing or escaped output root",
    )
    original, refs, actual_inputs, stat_only, lifecycle, cv2_refs, cases = evidence(config)
    sources = source_closure(original)
    snapshots = snapshot_closure(original, root, sources)
    (root / "preparation" / "source_snapshot").mkdir(parents=True, exist_ok=False)
    new_paths = sorted(set(sources) - set(original["source_sha256"]))
    for index, source in enumerate(new_paths):
        destination = root / "preparation" / "source_snapshot" / f"{index:04d}_{Path(source).name}"
        shutil.copyfile(source, destination)
        require(sha256(destination) == sources[source], "new source changed while snapshotting")
    result = build(config, original, refs, actual_inputs, stat_only, lifecycle, cv2_refs, cases, sources, snapshots)
    path = root / "preparation" / "prepared.json"
    write_once(path, result)
    read_prepared(path, sha256(path))
    return path


def read_prepared(path, expected_sha):
    """Rebuild this literal continuation contract; never call old dynamic guards."""
    cpu_guard()
    path = Path(path).resolve()
    prepared = checked_json(path, expected_sha, {})
    config = checked_json(CONFIG, CONFIG_SHA, {})
    assert_config(config)
    root = (REPO / config["output_dir"]).resolve()
    require(
        root.is_relative_to(REPO / "outputs") and path == root / "preparation" / "prepared.json",
        "continuation prepared path escaped registered root",
    )
    require(sha256(PLAN) == PLAN_SHA, "continuation plan changed")
    original, refs, actual_inputs, stat_only, lifecycle, cv2_refs, cases = evidence(config)
    sources = source_closure(original)
    snapshots = snapshot_closure(original, root, sources)
    expected = build(config, original, refs, actual_inputs, stat_only, lifecycle, cv2_refs, cases, sources, snapshots)
    require(prepared == expected, "continuation schema/refs/roster/authorization/source contract changed")
    for source, digest in snapshots.items():
        require(sha256(source) == digest, f"continuation snapshot changed: {source}")
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    path = prepare(args.config, seed=args.seed)
    print(json.dumps(dict(prepared=str(path), sha256=sha256(path), cached_quality_verified=False, new_physics_steps=0)))


if __name__ == "__main__":
    main()
