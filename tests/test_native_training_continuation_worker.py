"""Continuation scheduling/startup fixtures, without HDF5 or physical stepping."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from s2a.artifacts import REPO
from s2a.native_environment import EnvironmentAdmissionError, environment_snapshot


def worker():
    path = REPO / "scripts/collect_wam_native_training_continuation.py"
    spec = importlib.util.spec_from_file_location("native_continuation_worker_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    module = worker()
    monkeypatch.chdir(REPO)
    for key, value in dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1").items():
        monkeypatch.setenv(key, value)
    records = [
        dict(name=f"task{task}_case{index}", task_index=task, native_steps=142 if task == 1 else 143 + (index >= 8))
        for task in range(3)
        for index in range(10)
    ]
    original_path = tmp_path / "original.json"
    original_path.write_text(json.dumps(dict(schema_version="s2a.native-training-preparation.v1", records=records)))
    original_ref = module.reference(original_path)
    prepared = dict(
        schema_version="s2a.native-training-continuation-preparation.v1",
        seed=42,
        config=dict(new_physics_steps=3652),
        output_root=str(tmp_path / "new_output"),
        original_prepared_ref=original_ref,
        records=records,
        remaining_records=[record for record in records if record["task_index"] in (1, 2)],
        cached_records=[record for record in records if record["task_index"] == 0],
        cv2_probe_ref=dict(fixture=True),
        cv2_runtime_source_refs=[dict(fixture=True)],
    )
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared))
    prepared_ref = module.reference(prepared_path)
    reads, imports = [], []

    def read_prepared(path, expected):
        assert path == prepared_path and expected == prepared_ref["sha256"]
        reads.append(str(path))
        return prepared

    def environment_admission(probe, sources):
        assert probe == prepared["cv2_probe_ref"] and sources == prepared["cv2_runtime_source_refs"]
        assert not (Path(prepared["output_root"]) / "collection/task1/worker_started.json").exists()
        imports.append("admitted-before-startup")
        return dict(
            schema_version="s2a.native-training-environment-readiness.v1",
            complete=True,
            passed=True,
            restore_performed=True,
            before=environment_snapshot(),
            after_import_cv2=dict(environment_snapshot(), LD_LIBRARY_PATH="fixture-known-loader-change"),
            after_restore=environment_snapshot(),
            fixture_only=True,
        )

    monkeypatch.setitem(
        sys.modules,
        "prepare_wam_native_training_continuation",
        SimpleNamespace(read_prepared=read_prepared),
    )
    monkeypatch.setattr(module, "restore_cv2_environment", environment_admission)
    args = SimpleNamespace(prepared=str(prepared_path), prepared_sha256=prepared_ref["sha256"], task_index=1, seed=42)
    return SimpleNamespace(
        module=module,
        prepared=prepared,
        prepared_path=prepared_path,
        original_ref=original_ref,
        args=args,
        imports=imports,
        reads=reads,
    )


def test_only_remaining_tasks_and_exact_original_twenty_case_budget(continuation):
    item = continuation
    original = json.loads(Path(item.original_ref["path"]).read_text())
    for task in (1, 2):
        selected = item.module.task_records(item.prepared, original, task)
        assert len(selected) == 10 and all(record["task_index"] == task for record in selected)
        assert not any(record in item.prepared["cached_records"] for record in selected)
    assert sum(record["native_steps"] + 40 for record in item.prepared["remaining_records"]) == 3652
    with pytest.raises(ValueError, match="cached task0"):
        item.module.task_records(item.prepared, original, 0)
    changed = dict(item.prepared, remaining_records=list(reversed(item.prepared["remaining_records"])))
    with pytest.raises(ValueError, match="original fixed source roster"):
        item.module.task_records(changed, original, 1)
    changed = dict(item.prepared, config=dict(new_physics_steps=3653))
    with pytest.raises(ValueError, match="physics budget differs"):
        item.module.task_records(changed, original, 1)


def test_real_run_startup_follows_environment_readiness_and_preserves_original_source(continuation, monkeypatch):
    item, observed = continuation, []

    def stop_before_physics(original, original_ref, record, directory, progress):
        started_path = directory.parent / "worker_started.json"
        started = json.loads(started_path.read_text())
        readiness = json.loads(Path(started["environment_readiness_ref"]["path"]).read_text())
        assert readiness["restore_performed"] is readiness["passed"] is True
        assert started["cuda"] == started["ld"] == ""
        assert started["prepared_ref"] == original_ref == item.original_ref
        assert started["continuation_prepared_ref"] == item.module.reference(item.prepared_path)
        assert started["pid"] == os.getpid() and started["ppid"] == os.getppid()
        assert original["records"] == item.prepared["records"] and record["task_index"] == 1
        assert progress == dict(environment_initializations=0, physics_steps=0, current_canvas_images=0)
        observed.append(record["name"])
        raise RuntimeError("intentional fixture stop before any native physics")

    monkeypatch.setattr(item.module, "collect_case", stop_before_physics)
    assert item.module.run(item.args) == 1
    destination = Path(item.prepared["output_root"]) / "collection/task1"
    result = json.loads((destination / "result.json").read_text())
    assert result["complete"] is result["passed"] is False and result["cases"] == []
    assert result["prepared_ref"] == item.original_ref
    assert result["continuation_prepared_ref"] == item.module.reference(item.prepared_path)
    assert item.imports == ["admitted-before-startup"] and observed == ["task1_case0"]
    assert result["environment_initializations"] == result["physics_steps"] == result["current_canvas_images"] == 0
    pair = result["case_environment"][0]
    assert pair["name"] == "task1_case0"
    for phase in ("before", "after"):
        receipt = json.loads(Path(pair[f"{phase}_ref"]["path"]).read_text())
        assert receipt["passed"] is True and receipt["phase"] == f"{phase}_case"
        assert receipt["actual_environment"] == environment_snapshot()
        assert receipt["bytecode"] == "1"
    with pytest.raises(FileExistsError):
        item.module.run(item.args)
    assert len(observed) == 1  # Existing output never restarts even a fixture case.


def test_unknown_post_case_environment_is_saved_and_not_reset_or_retried(continuation, monkeypatch):
    item, observed = continuation, []

    def mutate_and_stop(original, original_ref, record, directory, progress):
        observed.append(record["name"])
        monkeypatch.setenv("LD_LIBRARY_PATH", "unregistered-after-case")
        raise RuntimeError("fixture stopped without physical stepping")

    monkeypatch.setattr(item.module, "collect_case", mutate_and_stop)
    assert item.module.run(item.args) == 1
    result_path = Path(item.prepared["output_root"]) / "collection/task1/result.json"
    result = json.loads(result_path.read_text())
    assert result["complete"] is result["passed"] is False and observed == ["task1_case0"]
    after = json.loads(Path(result["case_environment"][0]["after_ref"]["path"]).read_text())
    assert after["passed"] is False and after["actual_environment"]["LD_LIBRARY_PATH"] == "unregistered-after-case"
    assert os.environ["LD_LIBRARY_PATH"] == "unregistered-after-case"
    assert result["failure_environment"]["LD_LIBRARY_PATH"] == "unregistered-after-case"
    assert result["physics_steps"] == result["environment_initializations"] == 0


def test_import_rejection_saves_failure_and_never_publishes_startup_or_collects(continuation, monkeypatch):
    item = continuation
    failed_receipt = dict(complete=False, passed=False, restore_performed=False, fixture_only=True)

    def rejected_import(*args):
        raise EnvironmentAdmissionError("unknown import mutation", failed_receipt)

    def forbidden_collect(*args):
        raise AssertionError("rejected import cannot enter collector")

    monkeypatch.setattr(item.module, "restore_cv2_environment", rejected_import)
    monkeypatch.setattr(item.module, "collect_case", forbidden_collect)
    assert item.module.run(item.args) == 1
    directory = Path(item.prepared["output_root"]) / "collection/task1"
    assert not (directory / "worker_started.json").exists()
    result = json.loads((directory / "result.json").read_text())
    assert json.loads(Path(result["environment_readiness_ref"]["path"]).read_text()) == failed_receipt
    assert result["cases"] == result["case_environment"] == []
    assert result["physics_steps"] == result["environment_initializations"] == 0


def test_real_cli_excludes_cached_task0_and_does_not_import_simulator():
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    help_result = subprocess.run(
        [sys.executable, "scripts/collect_wam_native_training_continuation.py", "--help"],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0 and "{1,2}" in help_result.stdout and "--prepared-sha256" in help_result.stdout
    rejected = subprocess.run(
        [
            sys.executable,
            "scripts/collect_wam_native_training_continuation.py",
            "--prepared",
            "does-not-exist",
            "--prepared-sha256",
            "not-read",
            "--task-index",
            "0",
            "--seed",
            "42",
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2 and "invalid choice" in rejected.stderr
