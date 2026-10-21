"""Continuation evidence fixtures: failed cache, environment and full denominator."""

import ast
import copy
import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_wam_native_training_continuation.py"
    spec = importlib.util.spec_from_file_location("native_continuation_independent_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(checker, path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    return checker.reference(path)


def lifecycle(checker, root, *, returncode=0, pid=300):
    command = ["python", "scripts/collect_wam_native_training_continuation.py", "--task-index", "1"]
    launch = dict(
        automatic_retry=False,
        child=dict(pid=pid, start_ticks=900 + pid),
        selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        command=command,
    )
    ended = dict(actual_wait=True, automatic_retry=False, returncode=returncode)
    release = dict(
        pid=pid,
        start_ticks=900 + pid,
        returncode=returncode,
        reaped=True,
        signal_sent=False,
        original_error=None if returncode == 0 else "ChildProcessError('failed with1')",
        only_owned_child_signalled=True,
    )
    return {
        key: write_json(checker, root / f"{key}.json", payload)
        for key, payload in zip(("launch", "exit", "release"), (launch, ended, release), strict=True)
    }


def failed_origin(checker, tmp_path, *, returncode=1, count=10, task_passed=False, outer_exit=1):
    root = tmp_path / "original"
    records = [dict(name=f"cached_{i}") for i in range(10)]
    original = dict(output_root=str(root), records=records)
    original_ref = write_json(checker, root / "preparation/prepared.json", original)
    cases = [
        write_json(checker, root / "collection/task0" / r["name"] / "result.json", dict(saved=True)) for r in records
    ]
    peer_ref = write_json(checker, root / "review/preparation_verified.json", dict(passed=True))
    task = dict(
        schema_version="s2a.native-training-task.v1",
        task_index=0,
        seed=42,
        prepared_ref=original_ref,
        complete=task_passed,
        passed=task_passed,
        error=dict(
            type="ValueError",
            message="030 source guard requires S2A cwd and empty CUDA/LD",
            traceback="read_prepared original traceback",
        ),
        cases=cases[:count],
        environment_initializations=10,
        physics_steps=1328,
        current_canvas_images=50,
        wam_initializations=0,
        denoise_calls=0,
        fits=0,
        fresh_cases=0,
        future_rgb_frames=0,
    )
    task_ref = write_json(checker, root / "collection/task0/result.json", task)
    life = lifecycle(checker, root / "orchestration/collection/task0", returncode=returncode)
    owner = dict(
        schema_version="s2a.native-training-owner.v1",
        complete=False,
        passed=False,
        prepared_ref=original_ref,
        source_peer_ref=peer_ref,
        automatic_retry=False,
        owned_commands=[],
        command_attempts=[dict(task_index=0, phase="collection", observed_lifecycle=life)],
        error=dict(type="ChildProcessError", message="025 command failed with1; no automatic retry"),
    )
    owner_ref = write_json(checker, root / "orchestration/collection/result.json", owner)
    tool = dict(
        schema_version="s2a.native-training-owner-tool-exit-observation.v1",
        original_owner_result_ref=owner_ref,
        original_task_result_ref=task_ref,
        prepared_ref=original_ref,
        task_release_ref=life["release"],
        actual_tool_terminal_exit_code=outer_exit,
        actual_task_wait_returncode=1,
        actual_task_reaped=True,
        owner_birth_absent=True,
        task_birth_absent=True,
        actual_task_signal_sent=False,
        root_manual_popen_wait_claimed=False,
        complete_case_refs=cases[:count],
        actual_physics_steps=1328,
        actual_current_canvases=50,
        complete_source_guard_passed=False,
        full30_collection_certified=False,
        independent_case_quality_verified=False,
        prior409_source_bytes_unchanged=True,
        new_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    prepared = dict(
        original_prepared_ref=original_ref,
        original_preparation_verification_ref=peer_ref,
        failed_task_ref=task_ref,
        failed_owner_ref=owner_ref,
        tool_exit_ref=write_json(checker, tmp_path / "tool.json", tool),
        original_lifecycle_refs=life,
        cached_case_refs=cases[:count],
    )
    return prepared, original


def test_failed_exit1_cache_is_auditable_without_rewriting_original_success(checker, tmp_path):
    prepared, original = failed_origin(checker, tmp_path)
    task, child = checker.verify_original_failure(prepared, original, checker.Evidence())
    assert task["complete"] is task["passed"] is False
    assert child["returncode"] == 1 and child["reaped"] and not child["signal_sent"]
    assert not (Path(original["output_root"]) / "collection/result.json").exists()


@pytest.mark.parametrize("change", [dict(returncode=0), dict(count=9), dict(task_passed=True), dict(outer_exit=0)])
def test_cached_prefix_rejects_false_normal_exit_partial_or_retroactive_pass(checker, tmp_path, change):
    prepared, original = failed_origin(checker, tmp_path, **change)
    with pytest.raises(ValueError):
        checker.verify_original_failure(prepared, original, checker.Evidence())


def test_original_remaining_start_or_public_success_refuses_continuation(checker, tmp_path):
    prepared, original = failed_origin(checker, tmp_path)
    forbidden = Path(original["output_root"]) / "collection/task1"
    forbidden.mkdir()
    with pytest.raises(ValueError, match="already started"):
        checker.verify_original_failure(prepared, original, checker.Evidence())
    forbidden.rmdir()
    write_json(checker, Path(original["output_root"]) / "collection/result.json", dict(passed=True))
    with pytest.raises(ValueError, match="public success"):
        checker.verify_original_failure(prepared, original, checker.Evidence())


def probe_fixture(checker, root):
    loader = root / "outputs/libero_env311/lib/python3.11/site-packages/cv2/__init__.py"
    refs = []
    for name in ("__init__.py", "config.py", "config-3.py"):
        path = loader.parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source " + name)
        refs.append(checker.reference(path))
    before = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(root))
    probe = dict(
        schema_version="s2a.native-training-import-environment-probe.v1",
        before=before,
        after_import_cv2=dict(before, LD_LIBRARY_PATH=str(loader.parent / "../../lib64") + ":"),
        cv2_loader_ref=refs[0],
        cv2_version="fixture4.11",
        reproduces_empty_ld_guard_rejection=True,
        original_failed_worker_end_environment_recorded=False,
        original_trigger_uniquely_identified=False,
        root_original_worker_replayed=False,
        new_environment_initializations=0,
        new_physics_steps=0,
        new_model_calls=0,
        new_hdf_reads=0,
    )
    return probe, refs


def test_exact_import_probe_and_restore_reject_unknown_mutation(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    probe, refs = probe_fixture(checker, tmp_path)
    checker.verify_probe(probe, refs, checker.Evidence())
    unknown = copy.deepcopy(probe)
    unknown["after_import_cv2"]["CUDA_VISIBLE_DEVICES"] = "6"
    with pytest.raises(ValueError, match="registered cv2 mutation"):
        checker.verify_probe(unknown, refs, checker.Evidence())
    with pytest.raises(ValueError, match="strict CPU"):
        checker.strict_environment(probe["after_import_cv2"])
    assert probe["original_trigger_uniquely_identified"] is False


def test_literal423_source_closure_survives_json_sorting_and_rejects_extra_source(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    inherited, snapshots = {}, {}
    for index in range(409):
        for base, mapping in (("oldsource", inherited), ("oldsnapshot", snapshots)):
            path = tmp_path / base / f"{index}.py"
            path.parent.mkdir(exist_ok=True)
            path.write_text(f"# frozen {index}\n")
            mapping[str(path)] = checker.sha256(path)
    sources = dict(inherited)
    for relative in checker.NEW_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# new " + relative)
        sources[str(path)] = checker.sha256(path)
    root = tmp_path / "outputs/new"
    combined = dict(snapshots)
    for index, source in enumerate(sorted(set(sources) - set(inherited))):
        target = root / "preparation/source_snapshot" / f"{index:04d}_{Path(source).name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
        combined[str(target)] = checker.sha256(target)
    original = dict(source_sha256=inherited, snapshot_sha256=snapshots)
    prepared = json.loads(
        json.dumps(
            dict(
                source_sha256=sources,
                snapshot_sha256=combined,
                inherited_source_sha256=inherited,
                inherited_snapshot_sha256=snapshots,
                output_root=str(root),
            ),
            sort_keys=True,
        )
    )
    checker.source_closure(prepared, original, checker.Evidence())
    prepared["source_sha256"][str(tmp_path / "unexpected.py")] = "0" * 64
    with pytest.raises(ValueError, match="literal409"):
        checker.source_closure(prepared, original, checker.Evidence())


def readiness_fixture(checker, prepared, probe, refs, directory, *, pid):
    life = lifecycle(checker, directory / "owned", pid=pid)
    launch = json.loads(Path(life["launch"]["path"]).read_text())
    actual = dict(
        schema_version="s2a.native-training-environment-readiness.v1",
        complete=True,
        passed=True,
        restore_performed=True,
        continuation_prepared_ref=checker.reference(Path(prepared["output_root"]) / "preparation/prepared.json"),
        cv2_probe_ref=prepared["cv2_probe_ref"],
        cv2_runtime_source_refs=refs,
        before=probe["before"],
        after_import_cv2=probe["after_import_cv2"],
        after_restore=probe["before"],
        source_before_sha256={r["path"]: r["sha256"] for r in refs},
        source_after_sha256={r["path"]: r["sha256"] for r in refs},
        actual_cv2_loader=probe["cv2_loader_ref"]["path"],
        actual_cv2_loader_ref=probe["cv2_loader_ref"],
        actual_cv2_version=probe["cv2_version"],
        bytecode_before="1",
        bytecode_after_import="1",
        bytecode_after_restore="1",
        pid=pid,
        argv=launch["command"][1:],
        new_environment_initializations=0,
        new_physics_steps=0,
        new_model_calls=0,
        new_hdf_reads=0,
    )
    return actual, life, launch


def test_actual_readiness_requires_restore_before_startup_and_owned_process(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    probe, refs = probe_fixture(checker, tmp_path)
    root = tmp_path / "outputs/new"
    prepared = dict(
        output_root=str(root),
        cv2_runtime_source_refs=refs,
        cv2_probe_ref=write_json(checker, tmp_path / "probe.json", probe),
    )
    write_json(checker, root / "preparation/prepared.json", prepared)
    actual, _, launch = readiness_fixture(checker, prepared, probe, refs, root, pid=333)
    path = root / "readiness.json"
    checker.verify_readiness(write_json(checker, path, actual), prepared, launch, checker.Evidence())
    for key, value in (("after_restore", probe["after_import_cv2"]), ("pid", 999), ("restore_performed", False)):
        wrong = dict(actual, **{key: value})
        with pytest.raises(ValueError, match="admission"):
            checker.verify_readiness(write_json(checker, path, wrong), prepared, launch, checker.Evidence())


def collection_fixture(checker, tmp_path, monkeypatch, *, failed_case=False):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    root, oldroot = tmp_path / "outputs/new", tmp_path / "outputs/old"
    probe, runtime = probe_fixture(checker, tmp_path)
    records, case_refs, calls = [], [], []
    lengths = [[92] * 9 + [100], [180] * 9 + [194], [100] * 9 + [138]]
    for task in range(3):
        for index, n in enumerate(lengths[task]):
            name = f"task{task}_demo{index}"
            expert = oldroot / "sources" / name / "expert.npy"
            expert.parent.mkdir(parents=True, exist_ok=True)
            np.save(expert, np.full((n, 7), task + 0.25, np.float64), allow_pickle=False)
            records.append(
                dict(
                    name=name,
                    task_index=task,
                    native_steps=n,
                    expert_path=str(expert),
                    expert_sha256=checker.sha256(expert),
                    expert_dtype="float64",
                )
            )
    original = dict(output_root=str(oldroot), records=records)
    original_ref = write_json(checker, oldroot / "preparation/prepared.json", original)
    for record in records:
        directory = oldroot if record["task_index"] == 0 else root
        case_refs.append(
            write_json(
                checker,
                directory / "collection" / f"task{record['task_index']}" / record["name"] / "result.json",
                dict(name=record["name"], prepared_ref=original_ref, passed=True),
            )
        )
    prepared = dict(
        output_root=str(root),
        original_prepared_ref=original_ref,
        records=records,
        cached_records=records[:10],
        remaining_records=records[10:],
        cv2_runtime_source_refs=runtime,
        cv2_probe_ref=write_json(checker, tmp_path / "probe.json", probe),
        failed_task_ref=write_json(checker, oldroot / "collection/task0/result.json", dict(passed=False)),
    )
    prepared_ref = write_json(checker, root / "preparation/prepared.json", prepared)

    def case_contract(actual, record, source, native, evidence):
        assert source is original and actual["prepared_ref"] == original_ref
        np.testing.assert_array_equal(native, np.full((record["native_steps"], 7), record["task_index"] + 0.25))
        calls.append(record["name"])
        passed = not (failed_case and record is records[24])
        return dict(
            name=record["name"],
            task_index=record["task_index"],
            native_steps=record["native_steps"],
            physics_steps=record["native_steps"] + 40,
            stable_native_success=passed,
        )

    frozen = checker.load_gold(checker.Evidence())
    gold = SimpleNamespace(case_contract=case_contract, quality_summary=frozen.quality_summary)
    monkeypatch.setattr(checker, "case_contract", case_contract)
    old_task = dict(cases=case_refs[:10])
    prefix = checker.verify_prefix(prepared, original, old_task, dict(returncode=1), checker.Evidence(), gold)
    cached_bytes = oldroot / "collection/task0/cached_array_bytes.npz"
    cached_bytes.write_bytes(b"saved cached data, never decoded by final summary reuse")
    prefix.update(
        schema_version="s2a.native-training-continuation-independent-verification.v1",
        phase="prefix",
        passed=True,
        prepared=prepared_ref,
        input_sha256={prepared_ref["path"]: prepared_ref["sha256"], str(cached_bytes): checker.sha256(cached_bytes)},
        eligible_for_model_training_registration=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
    )
    prefix_path = root / "review/prefix_verified.json"
    prefix_ref = write_json(checker, prefix_path, prefix)
    tasks = []
    for task_index in (1, 2):
        directory = root / "collection" / f"task{task_index}"
        readiness, life, _ = readiness_fixture(checker, prepared, probe, runtime, directory, pid=300 + task_index)
        ready_ref = write_json(checker, directory / "environment_readiness.json", readiness)
        write_json(
            checker,
            directory / "worker_started.json",
            dict(
                pid=readiness["pid"],
                argv=readiness["argv"],
                environment_readiness_ref=ready_ref,
                prepared_ref=original_ref,
                continuation_prepared_ref=prepared_ref,
                task_index=task_index,
                cuda="",
                ld="",
                actual_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
            ),
        )
        environments = []
        for record in records[10 * task_index : 10 * (task_index + 1)]:
            item = dict(name=record["name"])
            for suffix in ("before", "after"):
                payload = dict(
                    schema_version="s2a.native-training-case-environment.v1",
                    name=record["name"],
                    phase=suffix + "_case",
                    continuation_prepared_ref=prepared_ref,
                    passed=True,
                    bytecode="1",
                    pid=readiness["pid"],
                    argv=readiness["argv"],
                    actual_environment=probe["before"],
                )
                item[suffix + "_ref"] = write_json(
                    checker, directory / "environment" / f"{record['name']}_{suffix}.json", payload
                )
            environments.append(item)
        result = dict(
            schema_version="s2a.native-training-continuation-task.v1",
            complete=True,
            passed=True,
            task_index=task_index,
            seed=42,
            prepared_ref=original_ref,
            continuation_prepared_ref=prepared_ref,
            actual_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
            environment_initializations=10,
            current_canvas_images=50,
            automatic_retry=False,
            case_environment=environments,
            cached_cases_recollected=0,
            wam_initializations=0,
            denoise_calls=0,
            fits=0,
            fresh_cases=0,
            future_rgb_frames=0,
            source_hdf5_reads=0,
            environment_readiness_ref=ready_ref,
            cases=case_refs[10 * task_index : 10 * (task_index + 1)],
            physics_steps=sum(n + 40 for n in lengths[task_index]),
            native_quality_passed=not (failed_case and task_index == 2),
        )
        tasks.append(
            dict(task_index=task_index, result=write_json(checker, directory / "result.json", result), lifecycle=life)
        )
    public = dict(
        schema_version="s2a.native-training-continuation-collection.v1",
        complete=True,
        passed=True,
        prepared_ref=prepared_ref,
        original_prepared_ref=original_ref,
        prefix_verification_ref=prefix_ref,
        cached_task_ref=prepared["failed_task_ref"],
        environment_initializations=30,
        physics_steps=4980,
        current_canvas_images=150,
        new_environment_initializations=20,
        new_collection_physics_steps=3652,
        new_current_canvas_images=100,
        cached_cases=10,
        cached_physics_steps=1328,
        cached_current_canvas_images=50,
        automatic_retry=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        wam_initializations=0,
        denoise_calls=0,
        fits=0,
        fresh_cases=0,
        task_results=tasks,
    )
    public_path = root / "collection/result.json"
    write_json(checker, public_path, public)
    return prepared, original, old_task, gold, calls, prefix_path, public_path, public


def test_complete30_keeps_cached_failed_origin_and_only_audits_new20(checker, tmp_path, monkeypatch):
    prepared, original, old_task, gold, calls, prefix_path, public_path, _ = collection_fixture(
        checker, tmp_path, monkeypatch
    )
    assert len(calls) == 10
    calls.clear()
    result = checker.verify_collection(public_path, prefix_path, prepared, original, old_task, checker.Evidence(), gold)
    assert len(calls) == 20 and all(not name.startswith("task0") for name in calls)
    assert len(result["case_verification"]) == 30 and result["physics_steps"] == 4980
    assert result["native_quality"]["eligible_for_model_training_registration"] is True
    assert result["original_task_passed"] is result["original_owner_passed"] is False
    assert result["cached_case_artifacts_rescored"] is False


def test_native_quality_false_preserves30_and_does_not_fail_evidence_integrity(checker, tmp_path, monkeypatch):
    prepared, original, old_task, gold, _, prefix_path, public_path, _ = collection_fixture(
        checker, tmp_path, monkeypatch, failed_case=True
    )
    result = checker.verify_collection(public_path, prefix_path, prepared, original, old_task, checker.Evidence(), gold)
    assert result["evidence_integrity_passed"] is True
    assert result["native_quality"]["total_cases"] == 30 and result["native_quality"]["stable_native_cases"] == 29
    assert result["eligible_for_model_training_registration"] is False


def test_new_case_cannot_claim_continuation_prepared_as_original_label_origin(checker, tmp_path, monkeypatch):
    prepared, original, old_task, gold, _, prefix_path, public_path, public = collection_fixture(
        checker, tmp_path, monkeypatch
    )
    ref = public["task_results"][0]["result"]
    task = json.loads(Path(ref["path"]).read_text())
    task["prepared_ref"] = public["prepared_ref"]
    public["task_results"][0]["result"] = write_json(checker, Path(ref["path"]), task)
    write_json(checker, public_path, public)
    with pytest.raises(ValueError, match="new task origin"):
        checker.verify_collection(public_path, prefix_path, prepared, original, old_task, checker.Evidence(), gold)


def test_reusing_prefix_summary_rechecks_consumed_bytes_without_decoding_or_rescoring(checker, tmp_path, monkeypatch):
    prepared, original, old_task, gold, calls, prefix_path, public_path, _ = collection_fixture(
        checker, tmp_path, monkeypatch
    )
    calls.clear()
    cached = Path(original["output_root"]) / "collection/task0/cached_array_bytes.npz"
    cached.write_bytes(b"changed after prefix audit")
    with pytest.raises(ValueError, match="evidence SHA differs"):
        checker.verify_collection(public_path, prefix_path, prepared, original, old_task, checker.Evidence(), gold)
    assert not calls


def test_hdf_content_is_never_opened_and_audit_failure_is_saved_without_overwrite(checker, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="must not reopen HDF"):
        checker.Evidence().check(tmp_path / "unopened.hdf5", "0" * 64)
    monkeypatch.setattr(checker, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    output = tmp_path / "outputs/rejected.json"

    def reject(*args):
        raise ValueError("fixture source rejected")

    monkeypatch.setattr(checker, "verify_preparation", reject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--phase",
            "prefix",
            "--prepared",
            "absent.json",
            "--prepared-sha256",
            "0" * 64,
            "--output",
            str(output),
            "--seed",
            "42",
        ],
    )
    with pytest.raises(SystemExit, match="fixture source rejected"):
        checker.main()
    saved = output.read_bytes()
    assert json.loads(saved)["passed"] is False
    assert json.loads(saved)["model_training_authorized"] is False
    with pytest.raises(ValueError, match="refuses overwrite"):
        checker.main()
    assert output.read_bytes() == saved


def test_real_script_entry_binds_all_phase_helpers_before_main_io():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_wam_native_training_continuation.py"

    class StopBeforeIO(Exception):
        pass

    def trace(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "main" and frame.f_code.co_filename == str(path):
            assert all(
                callable(frame.f_globals[name])
                for name in (
                    "verify_preparation",
                    "verify_prefix",
                    "verify_collection",
                    "verified_prefix",
                    "case_environment",
                    "case_contract",
                    "verify_first_continuation",
                )
            )
            raise StopBeforeIO
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(StopBeforeIO):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.settrace(previous)


def test_late_consumed_input_change_revokes_native_registration(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("{}")
    preparation = dict(robot_model_contract={}, original_checker_ref={}, failed_continuation_prefix_ref={})
    monkeypatch.setattr(checker, "verify_preparation", lambda *args: (preparation, {}, {}, {}))
    monkeypatch.setattr(checker, "load_gold", lambda *args: None)
    monkeypatch.setattr(
        checker,
        "verify_collection",
        lambda *args: dict(
            eligible_for_model_training_registration=True, collection_verified=True, evidence_integrity_passed=True
        ),
    )

    def changed(*args):
        raise ValueError("consumed input changed after score")

    monkeypatch.setattr(checker.Evidence, "recheck", changed)
    output = tmp_path / "outputs/late_failure.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--phase",
            "collection",
            "--prepared",
            str(prepared_path),
            "--prepared-sha256",
            checker.sha256(prepared_path),
            "--prefix-verification",
            "prefix.json",
            "--collection",
            "collection.json",
            "--output",
            str(output),
            "--seed",
            "42",
        ],
    )
    with pytest.raises(SystemExit, match="consumed input changed"):
        checker.main()
    actual = json.loads(output.read_text())
    assert (
        actual["passed"] is actual["collection_verified"] is actual["eligible_for_model_training_registration"] is False
    )
    assert actual["model_training_authorized"] is actual["fresh_performance_authorized"] is False


def test_literal_case_contract_only_changes_actual_robot_class_constant(checker):
    original = ast.parse(checker.OLD_CHECKER.read_text())
    revised = ast.parse(Path(checker.__file__).read_text())
    expected = next(
        node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "case_contract"
    )
    actual = next(node for node in revised.body if isinstance(node, ast.FunctionDef) and node.name == "case_contract")
    changed = 0
    for node in ast.walk(expected):
        if isinstance(node, ast.Constant) and node.value == "Panda":
            node.value = "MountedPanda"
            changed += 1
    assert changed == 1
    assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)
    assert checker.sha256(checker.OLD_CHECKER) == checker.OLD_CHECKER_SHA


def test_actual_mounted_panda_case_runs_all_frozen_geometry_pixels_and_success(checker, tmp_path):
    gold = checker.load_gold(checker.Evidence())
    frozen_test_path = Path(__file__).with_name("test_native_training_verification.py")
    spec = importlib.util.spec_from_file_location("native_frozen_synthetic_geometry_fixture", frozen_test_path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    record, h, native, arrays = fixtures.trajectory_fixture(gold)
    expert_path = tmp_path / "expert.npy"
    np.save(expert_path, native, allow_pickle=False)
    record.update(
        name="mounted_synthetic", task_index=0, expert_path=str(expert_path), expert_sha256=checker.sha256(expert_path)
    )
    prepared = dict(
        output_root=str(tmp_path),
        tool_contract_ref=dict(path="synthetic tool", sha256="fixture"),
        tool_source_from_tcp=h.tolist(),
    )
    prepared_ref = write_json(checker, tmp_path / "preparation/prepared.json", prepared)
    arrays_path = tmp_path / "trajectory.npz"
    np.savez(arrays_path, **arrays)
    receipts = []
    for index in range(len(arrays["sim_states"])):
        offsets = arrays["gripper_current_action_offsets"]
        cache = arrays["gripper_current_action_flat"][offsets[index] : offsets[index + 1]]
        receipts.append(
            dict(
                endpoint_step=index,
                sim_time=float(arrays["sim_time"][index]),
                sim_state_sha256=gold.state_hash(arrays["sim_states"][index]),
                sim_state_dtype="float64",
                sim_state_shape=list(arrays["sim_states"][index].shape),
                native_success=bool(arrays["native_success"][index]),
                gripper_current_action_shape=list(cache.shape),
                gripper_current_action_dtype="float64",
                gripper_current_action_sha256=hashlib.sha256(cache.tobytes()).hexdigest(),
            )
        )
    endpoints = tmp_path / "endpoints.jsonl"
    endpoints.write_text("\n".join(json.dumps(receipt) for receipt in receipts) + "\n")
    anchor, *_ = fixtures.camera_fixture(gold, tmp_path)
    anchors = []
    for index in gold.anchors_for(record["native_steps"]):
        copied = copy.deepcopy(anchor)
        copied.update(
            frame_index=index,
            current_endpoint=receipts[index],
            current_source_pose=arrays["source_pose"][index].tolist(),
            current_opening=float(arrays["opening"][index]),
            render_before=dict(
                sim_time=receipts[index]["sim_time"], sim_state_sha256=receipts[index]["sim_state_sha256"]
            ),
            render_after=dict(
                sim_time=receipts[index]["sim_time"], sim_state_sha256=receipts[index]["sim_state_sha256"]
            ),
        )
        anchors.append(copied)
    observation = dict(
        source_position="robot0_eef_pos = actual grip-site position",
        source_rotation="robot0_eef_quat xyzw = actual robot EEF BODY orientation",
        controller_state="cached actual fields; no forced endpoint updates",
        actual_source_site_id=1,
        actual_eef_body_id=2,
        actual_source_site_name="grip_site",
        actual_eef_body_name="eef_body",
        controller_site_name="controller_site",
        tool_contract_ref=prepared["tool_contract_ref"],
        delta_convention="inverse(T_tcp[k]) @ T_tcp[k+1], local right multiply",
        units=dict(position="m", time="s", rotation="column rot6d"),
        labels_normalized=False,
        camera_observables=False,
        initial_seed=42,
        initial_trial=0,
        pre_rollout_steps=0,
        initial_controller_refreshed=True,
        source_from_tcp=h.tolist(),
    )
    controller = dict(
        control_freq=20,
        robot_model="MountedPanda",
        gripper_model="PandaGripper",
        controller_site="controller_site",
        control_frame="MuJoCo world",
        model_timestep=0.002,
        **{key: [1.0] * 6 for key in ("input_min", "input_max", "output_min", "output_max", "kp", "kd")},
    )
    actual = dict(
        schema_version="s2a.native-training-case.v1",
        complete=True,
        passed=True,
        record=record,
        prepared_ref=prepared_ref,
        native_steps=len(native),
        physics_steps=len(native) + 40,
        endpoint_count=len(native) + 41,
        environment_initializations=1,
        current_canvas_images=5,
        future_rgb_frames=0,
        source_hdf5_reads=0,
        stable_success_last_steps=40,
        native_command_ref=dict(path=str(expert_path), sha256=record["expert_sha256"]),
        observation_contract=observation,
        controller_contract=controller,
        arrays=checker.reference(arrays_path),
        array_index={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()},
        original_dtypes={key: str(value.dtype) for key, value in arrays.items()},
        endpoint_receipts=checker.reference(endpoints),
        anchors=anchors,
        native_quality_passed=True,
    )
    result = checker.case_contract(actual, record, prepared, native, checker.Evidence())
    assert result["stable_native_success"] is True and result["maximum_canvas_channel_difference"] == 0
    assert result["maximum_anchor_reconstruction_error"]["position"] < 1e-12
    unchanged = copy.deepcopy(actual)
    with pytest.raises(ValueError, match="native robot/controller/site/clock"):
        gold.case_contract(actual, record, prepared, native, checker.Evidence())
    assert actual == unchanged
    wrong = copy.deepcopy(actual)
    wrong["controller_contract"]["robot_model"] = "Panda"
    with pytest.raises(ValueError, match="native robot/controller/site/clock"):
        checker.case_contract(wrong, record, prepared, native, checker.Evidence())


def test_first_failed421_is_preserved_as_snapshots_and_only_four_sources_can_change(checker, tmp_path):
    original = dict(
        source_sha256={f"/fixture/original_{i}.py": f"{i:064x}" for i in range(409)},
        snapshot_sha256={f"/fixture/snap_{i}.py": f"{i:064x}" for i in range(409)},
        records=[dict(name="saved")],
    )
    first_sources = dict(original["source_sha256"])
    for index, relative in enumerate(checker.NEW_SOURCES[:-2]):
        first_sources[str(checker.REPO / relative)] = f"{500 + index:064x}"
    root = tmp_path / "first"
    snapshots = dict(original["snapshot_sha256"])
    for index, source in enumerate(sorted(set(first_sources) - set(original["source_sha256"]))):
        snapshots[str(root / "preparation/source_snapshot" / f"{index:04d}_{Path(source).name}")] = first_sources[
            source
        ]

    def ref(name):
        return dict(path=str(tmp_path / name), sha256="a" * 64)

    original_ref, first_ref, task_ref = ref("original.json"), ref("first.json"), ref("failed_task.json")
    previous_config = dict(path=str(checker.FIRST_CONFIG), sha256=checker.FIRST_CONFIG_SHA)
    previous_plan = dict(path=str(checker.FIRST_PLAN), sha256=checker.FIRST_PLAN_SHA)
    first = dict(
        schema_version="s2a.native-training-continuation-preparation.v1",
        output_root=str(root),
        config_ref=previous_config,
        registration_ref=previous_plan,
        original_prepared_ref=original_ref,
        original_prepared=original,
        records=original["records"],
        cached_quality_verified=False,
        eligible_for_model_training_registration=False,
        source_sha256=first_sources,
        snapshot_sha256=snapshots,
    )
    prefix = dict(
        schema_version="s2a.native-training-continuation-independent-verification.v1",
        phase="prefix",
        prepared=first_ref,
        passed=False,
        prefix_verified=False,
        evidence_integrity_passed=False,
        error=dict(type="ValueError", message="030 native robot/controller/site/clock differs"),
        model_training_authorized=False,
        fresh_performance_authorized=False,
    )
    life = {key: ref(key + ".json") for key in ("launch", "exit", "release")}
    launch = dict(
        selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        automatic_retry=False,
        child=dict(pid=123, start_ticks=456),
    )
    ended = dict(actual_wait=True, automatic_retry=False, returncode=1)
    release = dict(
        pid=123,
        start_ticks=456,
        returncode=1,
        reaped=True,
        signal_sent=False,
        only_owned_child_signalled=True,
        original_error="failed with1",
    )
    owner = dict(
        schema_version="s2a.native-training-continuation-owner.v1",
        complete=False,
        passed=False,
        prepared_ref=first_ref,
        original_prepared_ref=original_ref,
        cached_task_ref=task_ref,
        owned_commands=[],
        automatic_retry=False,
        command_attempts=[dict(phase="prefix_audit", task_index=None, observed_lifecycle=life)],
    )
    cached = [ref("cached_case.json")]
    robot_refs = [ref(f"robot_source_{i}.py") for i in range(6)]
    prepared = dict(
        first_continuation_prepared_ref=first_ref,
        original_prepared_ref=original_ref,
        first_continuation_source_sha256=first_sources,
        first_continuation_snapshot_sha256=snapshots,
        source_sha256=dict(first_sources),
        permitted_revision_sources=sorted(str(checker.REPO / p) for p in checker.REVISED_SOURCES),
        previous_continuation_config_ref=previous_config,
        previous_continuation_registration_ref=previous_plan,
        failed_task_ref=task_ref,
        first_continuation_lifecycle_refs=life,
        cached_case_refs=cached,
        failed_continuation_prefix_ref=ref("failed_prefix.json"),
        failed_continuation_owner_ref=ref("failed_owner.json"),
        original_checker_ref=dict(path=str(checker.OLD_CHECKER), sha256=checker.OLD_CHECKER_SHA),
        first_continuation_owner_passed=False,
        first_prefix_verified=False,
        robot_model_contract=dict(
            expected_actual_class="MountedPanda",
            actual_parent_class="ManipulatorModel",
            comparison_class="Panda",
            comparison_is_parent=False,
            rename_actual_evidence=False,
            accept_unmounted_panda=False,
        ),
        robot_runtime_source_refs=robot_refs,
        config=dict(robot_runtime_source_refs=robot_refs),
    )
    for path in prepared["permitted_revision_sources"]:
        prepared["source_sha256"][path] = "b" * 64
    observed = dict(
        schema_version="s2a.native-training-model-contract-failure-observation.v1",
        first_prepared_ref=first_ref,
        failed_prefix_ref=prepared["failed_continuation_prefix_ref"],
        failed_continuation_owner_ref=prepared["failed_continuation_owner_ref"],
        prefix_lifecycle_refs=life,
        actual_owner_tool_terminal_exit_code=1,
        actual_prefix_wait_returncode=1,
        actual_prefix_reaped=True,
        owner_pid_now_absent=True,
        prefix_pid_now_absent=True,
        actual_prefix_signal_sent=False,
        root_manual_popen_wait_claimed=False,
        prefix_pid=123,
        prefix_start_ticks=456,
        old10_case_refs=cached,
        first_case_ref=cached[0],
        old409_checker_ref=prepared["original_checker_ref"],
        original409_source_bytes_unchanged=True,
        first421_snapshot_bytes_unchanged=True,
        controller_predicate_checks=dict(
            control_freq=True,
            robot_model=False,
            gripper_model=True,
            controller_site=True,
            control_frame=True,
            model_timestep=True,
        ),
        expected_by_old_checker="Panda",
        actual_controller_robot_model="MountedPanda",
        new_environment_initializations=0,
        new_physics_steps=0,
        new_current_canvases=0,
        new_hdf_reads=0,
        new_model_calls=0,
        new_fits=0,
        fresh_cases=0,
        automatic_retry=False,
        full30_collection_certified=False,
        independent_case_quality_verified=False,
    )
    payloads = dict(
        first_continuation_prepared=first,
        failed_continuation_prefix=prefix,
        failed_continuation_owner=owner,
        model_contract_failure_observation=observed,
    )
    documents = {
        life[key]["path"]: payload
        for key, payload in zip(("launch", "exit", "release"), (launch, ended, release), strict=True)
    }
    checked = []
    evidence = SimpleNamespace(ref=lambda item: documents[item["path"]], check=lambda p, d: checked.append((p, d)))
    checker.verify_first_continuation(prepared, original, payloads, evidence)
    assert all((path, digest) in checked for path, digest in snapshots.items())
    wrong = copy.deepcopy(prepared)
    wrong["source_sha256"][str(checker.REPO / "scripts/collect_wam_native_training_continuation.py")] = "c" * 64
    with pytest.raises(ValueError, match="registered four"):
        checker.verify_first_continuation(wrong, original, payloads, evidence)
    wrong = dict(prepared, first_continuation_owner_passed=True)
    with pytest.raises(ValueError, match="mounted class proof"):
        checker.verify_first_continuation(wrong, original, payloads, evidence)
