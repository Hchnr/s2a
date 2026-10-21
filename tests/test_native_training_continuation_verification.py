"""Continuation evidence fixtures: failed cache, environment and full denominator."""

import copy
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


def test_literal421_source_closure_survives_json_sorting_and_rejects_extra_source(checker, tmp_path, monkeypatch):
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
    monkeypatch.setattr(checker, "verify_preparation", lambda *args: ({}, {}, {}, {}))
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
