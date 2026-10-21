"""Literal-source continuation admission; synthetic files, no HDF or physics."""

import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def preparer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prepare_wam_native_training_continuation")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


@pytest.fixture
def synthetic(preparer, tmp_path, monkeypatch):
    """A complete409/49/30 fixture, not a permissive mock of the source guard."""
    config = json.loads(preparer.CONFIG.read_text())
    monkeypatch.setattr(preparer, "REPO", tmp_path)
    config_path = tmp_path / "configs/wam_native_training_continuation_20261021.json"
    plan = tmp_path / "plans/v0.0.1/030_environment_continuation_20261021.md"
    monkeypatch.setattr(preparer, "CONFIG", config_path)
    monkeypatch.setattr(preparer, "PLAN", plan)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    for name in preparer.NEW_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic registered continuation source.\n")
    monkeypatch.setattr(preparer, "tracked_new_sources", lambda: set(preparer.NEW_SOURCES))
    oldroot = tmp_path / "outputs/wam_stability_20261021/synchronous_native_training"
    sources, snapshots = {}, {}
    for index in range(409):
        path = (
            tmp_path / "scripts/verify_wam_native_training.py"
            if index == 0
            else tmp_path / "frozen_original_src" / f"source{index}.py"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# original source {index}\n")
        sources[str(path)] = preparer.sha256(path)
        snapshot = oldroot / "preparation/source_snapshot" / f"{index:04d}_{path.name}"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        snapshots[str(snapshot)] = preparer.sha256(snapshot)
    inputs, records, stats = {}, [], {}
    steps = [[93] * 8 + [92] * 2, [142] * 10, [143] * 8 + [144] * 2]
    for task in range(3):
        hdf = tmp_path / f"source{task}.hdf5"
        hdf.write_text("Synthetic HDF identity; opening these bytes is forbidden.\n")
        inputs[str(hdf)] = preparer.sha256(hdf)
        stats[str(hdf)] = preparer.source_stat(hdf)
        for index in range(10):
            name = f"task{task}_native_demo_{index}"
            expert = oldroot / "sources" / name / "expert.npy"
            expert.parent.mkdir(parents=True)
            expert.write_bytes(b"synthetic selected command byte cache")
            inputs[str(expert)] = preparer.sha256(expert)
            records.append(
                dict(
                    name=name,
                    task_index=task,
                    source_hdf5=str(hdf),
                    expert_path=str(expert),
                    native_steps=steps[task][index],
                    anchor_frames=[0, 1, 2, 3, steps[task][index]],
                )
            )
    runtime = []
    for relative in preparer.F64_RUNTIME_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic F64 observation source.\n")
        ref = preparer.reference(path)
        runtime.append(ref)
        inputs[ref["path"]] = ref["sha256"]
    for index in range(13):
        path = save(tmp_path / "old_metadata" / f"meta{index}.json", {"metadata": index})
        inputs[str(path)] = preparer.sha256(path)
    assert len(inputs) == 49
    original = dict(
        schema_version="s2a.native-training-preparation.v1",
        seed=42,
        source_sha256=sources,
        snapshot_sha256=snapshots,
        input_sha256=inputs,
        source_stats=stats,
        runtime_source_refs=runtime,
        records=records,
        metadata_admission=True,
        native_success_verified=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        config={"simulation_python": "outputs/libero_env311/bin/python"},
        output_root=str(oldroot),
        policy_config={"control_fps": 20},
        tool_contract_ref={"path": "synthetic contract", "sha256": "synthetic"},
        tool_source_from_tcp=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        fresh_config_ref={},
        fresh_fixtures_ref={},
        historical_fixture_refs=[],
        selection={},
        forbidden_initial_state_sha256=[],
    )
    original_path = save(tmp_path / config["original_prepared"], original)
    original_ref = preparer.reference(original_path)
    peer = dict(
        schema_version="s2a.native-training-independent-verification.v1",
        phase="prepare",
        passed=True,
        preparation_verified=True,
        prepared=original_ref,
        input_sha256={original_ref["path"]: original_ref["sha256"]},
        model_training_authorized=False,
        fresh_performance_authorized=False,
    )
    peer_path = save(tmp_path / config["original_preparation_verification"], peer)
    case_refs = []
    for record in records[:10]:
        case = save(oldroot / "collection/task0" / record["name"] / "result.json", {"cached": record["name"]})
        case_refs.append(preparer.reference(case))
    task = dict(
        schema_version="s2a.native-training-task.v1",
        prepared_ref=original_ref,
        task_index=0,
        seed=42,
        complete=False,
        passed=False,
        cases=case_refs,
        environment_initializations=10,
        physics_steps=1328,
        current_canvas_images=50,
        error=dict(type="ValueError", message="030 source guard requires S2A cwd and empty CUDA/LD"),
    )
    task_path = save(tmp_path / config["failed_task"], task)
    lifecycle = {}
    parent = oldroot / "orchestration/collection/task0"
    launch = dict(
        child=dict(pid=123, start_ticks=456), selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
    )
    for name, data in (
        ("launch", launch),
        ("exit", dict(actual_wait=True, returncode=1)),
        ("release", dict(returncode=1, reaped=True, signal_sent=False, pid=123, start_ticks=456)),
    ):
        path = parent / ("ownership/release.json" if name == "release" else f"{name}.json")
        lifecycle[name] = preparer.reference(save(path, data))
    owner = dict(
        schema_version="s2a.native-training-owner.v1",
        complete=False,
        passed=False,
        prepared_ref=original_ref,
        source_peer_ref=preparer.reference(peer_path),
        automatic_retry=False,
        owned_commands=[],
        command_attempts=[dict(task_index=0, observed_lifecycle=lifecycle)],
    )
    owner_path = save(tmp_path / config["failed_owner"], owner)
    observed = dict(
        original_owner_result_ref=preparer.reference(owner_path),
        original_task_result_ref=preparer.reference(task_path),
        prepared_ref=original_ref,
        actual_tool_terminal_exit_code=1,
        root_manual_popen_wait_claimed=False,
        actual_task_wait_returncode=1,
        actual_task_reaped=True,
        actual_task_signal_sent=False,
        owner_birth_absent=True,
        task_birth_absent=True,
        complete_case_refs=case_refs,
        actual_physics_steps=1328,
        actual_current_canvases=50,
        complete_source_guard_passed=False,
        full30_collection_certified=False,
        independent_case_quality_verified=False,
        prior409_source_bytes_unchanged=True,
        task_release_ref=lifecycle["release"],
    )
    save(tmp_path / config["tool_exit"], observed)
    for relative in preparer.CV2_RUNTIME_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic cv2 source; never imported.\n")
    before = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(tmp_path))
    after = dict(
        before,
        LD_LIBRARY_PATH=str(tmp_path / "outputs/libero_env311/lib/python3.11/site-packages/cv2/../../lib64") + ":",
    )
    probe = dict(
        schema_version="s2a.native-training-import-environment-probe.v1",
        before=before,
        after_import_cv2=after,
        cv2_loader_ref=preparer.reference(tmp_path / preparer.CV2_RUNTIME_SOURCE_PATHS[0]),
        reproduces_empty_ld_guard_rejection=True,
        original_failed_worker_end_environment_recorded=False,
        original_trigger_uniquely_identified=False,
        new_environment_initializations=0,
        new_physics_steps=0,
        new_model_calls=0,
        new_hdf_reads=0,
        root_original_worker_replayed=False,
    )
    save(tmp_path / config["cv2_probe"], probe)
    for key, _ in preparer.REF_FIELDS:
        config[key + "_sha256"] = preparer.sha256(tmp_path / config[key])
    save(config_path, config)
    monkeypatch.setattr(preparer, "CONFIG_SHA", preparer.sha256(config_path))
    monkeypatch.setattr(preparer, "PLAN_SHA", preparer.sha256(plan))
    # A real content read of even a synthetic HDF file is a test failure.
    original_sha = preparer.sha256

    def no_hdf_hash(path):
        assert Path(path).suffix not in (".hdf5", ".h5"), "continuation must not read inherited HDF bytes"
        return original_sha(path)

    monkeypatch.setattr(preparer, "sha256", no_hdf_hash)
    return dict(
        config=config,
        config_path=config_path,
        original=original,
        task=task,
        owner=owner,
        observed=observed,
        probe=probe,
        oldroot=oldroot,
        refs={
            field: dict(path=str(tmp_path / config[key]), sha256=config[key + "_sha256"])
            for key, field in preparer.REF_FIELDS
        },
    )


def test_full_synthetic_continuation_roundtrip_preserves409_and_only_new12(preparer, synthetic):
    path = preparer.prepare(synthetic["config_path"])
    prepared = preparer.read_prepared(path, preparer.sha256(path))
    assert len(prepared["source_sha256"]) == len(prepared["snapshot_sha256"]) == 421
    assert prepared["original_prepared"] == synthetic["original"]
    assert len(prepared["records"]) == 30 and len(prepared["remaining_records"]) == 20
    assert [r["task_index"] for r in prepared["remaining_records"]] == [1] * 10 + [2] * 10
    assert not prepared["cached_quality_verified"] and not prepared["original_task_passed"]
    assert not prepared["eligible_for_model_training_registration"] and not prepared["model_training_authorized"]
    assert all(ref["path"] not in prepared["actual_input_sha256"] for ref in prepared["inherited_hdf5_refs_stat_only"])
    assert all(prepared["snapshot_sha256"][p] == h for p, h in synthetic["original"]["snapshot_sha256"].items())
    path.write_text(json.dumps(prepared, sort_keys=True))
    assert preparer.read_prepared(path, preparer.sha256(path)) == prepared
    with pytest.raises(ValueError, match="existing"):
        preparer.prepare(synthetic["config_path"])


def test_literal_closure_requires_tracked_twelve_without_dynamic_expansion(preparer, synthetic, monkeypatch):
    monkeypatch.setattr(preparer, "tracked_new_sources", lambda: set(preparer.NEW_SOURCES[:-1]))
    with pytest.raises(ValueError, match="twelve.*tracked"):
        preparer.source_closure(synthetic["original"])
    monkeypatch.setattr(preparer, "tracked_new_sources", lambda: set(preparer.NEW_SOURCES))
    unrelated = preparer.REPO / "scripts/unrelated_new.py"
    unrelated.write_text("# Newly tracked elsewhere; not a continuation source.\n")
    sources = preparer.source_closure(synthetic["original"])
    assert str(unrelated) not in sources
    oldpath = Path(next(iter(synthetic["original"]["source_sha256"])))
    oldpath.write_text("# changed old source\n")
    with pytest.raises(ValueError, match="original409"):
        preparer.original_guard(
            synthetic["original"],
            synthetic["refs"]["original_prepared_ref"],
            json.loads(Path(synthetic["refs"]["original_preparation_verification_ref"]["path"]).read_text()),
            {},
        )


@pytest.mark.parametrize("mutation", ("task_passed", "wait0", "signal", "second_task", "task1_present"))
def test_original_failure_never_becomes_success_or_replay_authorization(preparer, synthetic, mutation):
    task, owner, observed = (copy.deepcopy(synthetic[key]) for key in ("task", "owner", "observed"))
    if mutation == "task_passed":
        task["passed"] = True
    elif mutation == "wait0":
        observed["actual_task_wait_returncode"] = 0
    elif mutation == "signal":
        observed["actual_task_signal_sent"] = True
    elif mutation == "second_task":
        owner["command_attempts"].append(dict(task_index=1))
    else:
        (synthetic["oldroot"] / "collection/task1").mkdir()
    with pytest.raises(ValueError):
        preparer.failure_admission(synthetic["original"], synthetic["refs"], task, owner, observed, {})


@pytest.mark.parametrize(
    "field,value", (("CUDA_VISIBLE_DEVICES", "6"), ("LD_LIBRARY_PATH", "unknown"), ("cwd", "/elsewhere"))
)
def test_probe_accepts_only_the_exact_observed_cv2_effect(preparer, synthetic, field, value):
    probe = copy.deepcopy(synthetic["probe"])
    probe["after_import_cv2"][field] = value
    with pytest.raises(ValueError, match="exact cv2"):
        preparer.probe_admission(probe, {})


def test_read_guard_rejects_tampered_remaining_roster_and_stat(preparer, synthetic):
    path = preparer.prepare(synthetic["config_path"])
    prepared = json.loads(path.read_text())
    prepared["remaining_records"] = prepared["records"][:20]
    path.write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="contract changed"):
        preparer.read_prepared(path, preparer.sha256(path))
    hdf = Path(next(iter(synthetic["original"]["source_stats"])))
    hdf.write_text("Changed native source identity; no content read is needed.\n")
    with pytest.raises(ValueError, match="HDF stat"):
        preparer.evidence(synthetic["config"])


def test_cpu_budget_guards_and_real_cli_help(preparer, monkeypatch):
    config = json.loads(preparer.CONFIG.read_text())
    config["remaining_tasks"] = [0, 1, 2]
    with pytest.raises(ValueError, match="budget or task"):
        preparer.assert_config(config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    with pytest.raises(ValueError, match="empty CUDA/LD"):
        preparer.prepare()
    command = subprocess.run(
        [sys.executable, str(Path(preparer.__file__)), "--help"],
        cwd=Path(preparer.__file__).resolve().parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert command.returncode == 0 and "--config" in command.stdout and "--seed" in command.stdout
