"""030 identity selection and preparation boundaries using synthetic sources only."""

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from s2a.native_training_sources import (
    initial_state_hash,
    numeric_demo_order,
    select_training_sources,
    training_anchor_frames,
    validate_selected_shapes,
)


def candidate(index, task=0, steps=50):
    state = [task + 1.0, index + 0.25]
    xml = f"<mujoco model='task{task}_demo{index}'/>"
    return dict(
        demo=f"demo_{index}",
        initial_state=state,
        initial_state_sha256=initial_state_hash(state),
        model_xml=xml,
        model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        actions_shape=[steps, 7],
    )


@pytest.fixture
def preparer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prepare_wam_native_training")


def test_global_rng_numeric_order_then_identity_aliases():
    tasks = [[candidate(i, task) for i in reversed(range(16))] for task in range(2)]
    excluded = [{"demo_2"}, {"demo_11"}]
    forbidden = {tasks[0][6]["initial_state_sha256"]}  # demo9; exclude by physical identity.
    original = copy.deepcopy(tasks)
    selected, selection = select_training_sources(tasks, excluded, forbidden, count=4)
    rng = np.random.default_rng(42)
    seen = set(forbidden)
    for task_index in range(2):
        names = [f"demo_{i}" for i in range(16) if f"demo_{i}" not in excluded[task_index]]
        expected_order = [names[int(i)] for i in rng.permutation(len(names))]
        actual = selection["tasks"][task_index]
        assert actual["numeric_demo_order"] == names
        assert actual["permutation_demo_order"] == expected_order
        expected = []
        for name in expected_order:
            row = next(item for item in tasks[task_index] if item["demo"] == name)
            if row["initial_state_sha256"] not in seen:
                expected.append(name)
                seen.add(row["initial_state_sha256"])
            if len(expected) == 4:
                break
        assert [row["demo"] for row in selected[task_index]] == expected
    assert tasks == original
    assert select_training_sources(tasks, excluded, forbidden, count=4)[1] == selection
    assert selection["tasks"][1]["permutation_demo_order"] != [
        selection["tasks"][1]["numeric_demo_order"][int(i)] for i in np.random.default_rng(42).permutation(15)
    ]


def test_duplicate_initial_cannot_cross_task_training_split():
    a, b = candidate(0), candidate(0, 1)
    b["initial_state_sha256"] = a["initial_state_sha256"]
    with pytest.raises(ValueError, match="insufficient"):
        select_training_sources([[a], [b]], [set(), set()], set(), count=1)
    with pytest.raises(ValueError, match="seed"):
        select_training_sources([[a]], [set()], set(), count=1, seed=43)


def test_selection_does_not_replace_bad_native_length_or_shape():
    # With one selected source there is no opportunity to silently skip it.
    selected, selection = select_training_sources([[candidate(0, steps=42)]], [set()], set(), count=1)
    assert selection["tasks"][0]["selected_demo_ids"] == ["demo_0"]
    with pytest.raises(ValueError, match="replacement"):
        validate_selected_shapes(selected)
    selected[0][0]["actions_shape"] = [50, 10]
    with pytest.raises(ValueError, match="N×7"):
        validate_selected_shapes(selected)


def test_five_full_windows_include_terminal_without_overlap_of_anchor_ids():
    assert training_anchor_frames(43) == [0, 1, 2, 3, 43]
    assert training_anchor_frames(600) == [0, 186, 373, 560, 600]
    for n in (43, 44, 117, 600):
        frames = training_anchor_frames(n)
        assert len(set(frames)) == 5 and frames[-1] == n
        assert all(f + 41 <= n + 41 for f in frames)
    for n in (42, 601, True, 43.0):
        with pytest.raises(ValueError):
            training_anchor_frames(n)


def test_state_identity_and_canonical_numeric_names():
    a = np.array([0.25, -0.5], dtype=np.float32)
    assert initial_state_hash(a) == initial_state_hash(a.astype(np.float64))
    assert numeric_demo_order(["demo_12", "demo_2", "demo_0"]) == ["demo_0", "demo_2", "demo_12"]
    for names in (["demo_1", "demo_1"], ["demo_01"], ["obs"], [1]):
        with pytest.raises(ValueError):
            numeric_demo_order(names)
    for state in ([], [float("nan")], [[1.0]]):
        with pytest.raises(ValueError):
            initial_state_hash(state)


class SyntheticDataset:
    def __init__(self, values, accesses, name):
        self.values, self.accesses, self.name = values, accesses, name
        self.shape = values.shape

    def __getitem__(self, key):
        self.accesses.append((self.name, repr(key)))
        if self.name.endswith("states"):
            assert key == 0  # Never read a full source trajectory.
        return self.values[key]


class SyntheticDemo:
    def __init__(self, name, accesses):
        self.attrs = {"model_file": b"<mujoco/>"}
        self.datasets = dict(
            states=SyntheticDataset(np.array([[1.0, 2.0], [3.0, 4.0]]), accesses, name + "/states"),
            actions=SyntheticDataset(np.zeros((43, 7), dtype=np.float32), accesses, name + "/actions"),
        )

    def __getitem__(self, key):
        assert key in ("states", "actions"), "obs/images are forbidden"
        return self.datasets[key]


class SyntheticGroup(dict):
    def __getitem__(self, key):
        assert key != "demo_0", "excluded/fresh demo must not be opened"
        return super().__getitem__(key)


class SyntheticHandle:
    def __init__(self, group):
        self.group = group

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __getitem__(self, key):
        assert key == "data"
        return self.group


def test_metadata_accesses_only_nonexcluded_initial_xml_and_action_shape(preparer):
    accesses = []
    group = SyntheticGroup(demo_0=object(), demo_1=SyntheticDemo("demo_1", accesses))
    rows = preparer.metadata_sources("synthetic", {"demo_0"}, opener=lambda *_: SyntheticHandle(group))
    assert [r["demo"] for r in rows] == ["demo_1"]
    assert rows[0]["actions_shape"] == [43, 7]
    assert accesses == [("demo_1/states", "0")]


def test_selected_export_preserves_seven_command_columns_and_dtype(preparer, tmp_path):
    accesses = []
    demo = SyntheticDemo("demo_1", accesses)
    values = np.arange(43 * 7, dtype=np.float32).reshape(43, 7) / 100
    demo.datasets["actions"].values = values
    group = SyntheticGroup(demo_0=object(), demo_1=demo)
    path = tmp_path / "selected" / "expert.npy"
    records = [dict(source_hdf5="synthetic", demo="demo_1", native_steps=43, expert_path=str(path))]
    inputs = {}
    preparer.export_actions(records, inputs, opener=lambda *_: SyntheticHandle(group))
    saved = np.load(path, allow_pickle=False)
    assert saved.dtype == values.dtype and np.array_equal(saved, values)
    assert inputs[str(path)] == records[0]["expert_sha256"] == preparer.sha256(path)
    assert accesses == [("demo_1/actions", "slice(None, None, None)")]
    with pytest.raises(FileExistsError):
        preparer.export_actions(records, inputs, opener=lambda *_: SyntheticHandle(group))


def test_export_rejects_invalid_selected_commands_without_creating_file(preparer, tmp_path):
    accesses = []
    demo = SyntheticDemo("demo_1", accesses)
    demo.datasets["actions"].values[0, 0] = np.nan
    path = tmp_path / "bad" / "expert.npy"
    with pytest.raises(ValueError, match="invalid"):
        preparer.export_actions(
            [dict(source_hdf5="synthetic", demo="demo_1", native_steps=43, expert_path=str(path))],
            {},
            opener=lambda *_: SyntheticHandle({"demo_1": demo}),
        )
    assert not path.exists()


def test_source_closure_requires_new_files_and_preserves_inherited_sha(preparer, tmp_path, monkeypatch):
    path = tmp_path / "src" / "old.py"
    path.parent.mkdir()
    path.write_text("old=1\n")
    plan, config = tmp_path / "plan.md", tmp_path / "config.json"
    plan.write_text("plan\n")
    config.write_text("{}\n")
    monkeypatch.setattr(preparer, "REPO", tmp_path)
    monkeypatch.setattr(preparer, "PLAN", plan)
    monkeypatch.setattr(preparer, "CONFIG", config)
    monkeypatch.setattr(preparer.subprocess, "check_output", lambda *a, **k: "src/old.py\n")
    monkeypatch.setattr(preparer, "NEW_SOURCES", ("src/missing.py",))
    prior = {"source_sha256": {str(path): preparer.sha256(path)}}
    with pytest.raises(ValueError, match="incomplete"):
        preparer.source_closure(prior)
    monkeypatch.setattr(preparer, "NEW_SOURCES", ())
    closure = preparer.source_closure(prior)
    assert closure[str(path)] == prior["source_sha256"][str(path)]
    path.write_text("changed=1\n")
    with pytest.raises(ValueError, match="original111"):
        preparer.source_closure(prior)


def test_prepare_refuses_existing_output_before_hdf_access(preparer, tmp_path, monkeypatch):
    config = json.loads(preparer.CONFIG.read_text())
    config["output_dir"] = "outputs/existing"
    output = tmp_path / config["output_dir"]
    output.mkdir(parents=True)
    config_path, plan = tmp_path / "config.json", tmp_path / "plan.md"
    config_path.write_text(json.dumps(config))
    plan.write_text("test\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(preparer, "REPO", tmp_path)
    monkeypatch.setattr(preparer, "CONFIG", config_path)
    monkeypatch.setattr(preparer, "CONFIG_SHA", preparer.sha256(config_path))
    monkeypatch.setattr(preparer, "PLAN", plan)
    monkeypatch.setattr(preparer, "PLAN_SHA", preparer.sha256(plan))
    monkeypatch.setattr(
        preparer, "prior_admission", lambda *a: pytest.fail("no source or HDF reads before output guard")
    )
    with pytest.raises(ValueError, match="existing"):
        preparer.prepare(config_path)


def test_cpu_guard_and_real_cli_help_before_any_dataset_read(preparer, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(ValueError, match="empty CUDA/LD"):
        preparer.prepare()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    command = subprocess.run(
        [sys.executable, str(Path(preparer.__file__)), "--help"],
        cwd=Path(preparer.__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert command.returncode == 0 and "--config" in command.stdout and "--seed" in command.stdout


@pytest.fixture
def synthetic_preparation(preparer, tmp_path, monkeypatch):
    """Exercise the real preparation/public-reader orchestration with fake HDF APIs."""
    config = json.loads(preparer.CONFIG.read_text())
    config["output_dir"] = "outputs/native"
    config["policy_template"] = "configs/policy.json"
    config_path, plan = tmp_path / "config.json", tmp_path / "plan.md"
    config_path.write_text(json.dumps(config))
    plan.write_text("synthetic plan\n")
    source = tmp_path / "source.py"
    source.write_text("source=1\n")
    runtime = tmp_path / "runtime_transform_utils.py"
    runtime.write_text("# Synthetic pinned simulator numerical source.\n")
    monkeypatch.setattr(preparer, "RUNTIME_SOURCE_PATHS", (runtime.name,))
    policy_path = tmp_path / "configs" / "policy.json"
    policy_path.parent.mkdir()
    tool_path = tmp_path / "tool.json"
    tool_path.write_text(json.dumps({"source_from_tcp": np.eye(4).tolist()}))
    policy_path.write_text(json.dumps(dict(tool_contract=str(tool_path))))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    for key, value in dict(
        REPO=tmp_path,
        CONFIG=config_path,
        CONFIG_SHA=preparer.sha256(config_path),
        PLAN=plan,
        PLAN_SHA=preparer.sha256(plan),
    ).items():
        monkeypatch.setattr(preparer, key, value)
    refs = {}
    for name in (
        "prior_diagnostic_verification_ref",
        "prior_diagnostic_final_peer_ref",
        "prior_prepared_ref",
        "fresh_config_ref",
        "fresh_fixtures_ref",
    ):
        path = tmp_path / (name + ".json")
        path.write_text("{}\n")
        refs[name] = preparer.reference(path)
    inherited = {str(source): preparer.sha256(source)}

    def prior_admission(_, inputs):
        result = {k: refs[k] for k in refs if k.startswith("prior_")}
        for ref in result.values():
            preparer.checked_json(ref["path"], ref["sha256"], inputs)
        return result, dict(source_sha256=inherited)

    task_sources, source_rows, groups = [], {}, {}
    for task in range(3):
        path = tmp_path / f"native{task}.hdf5"
        path.write_text("Synthetic bytes, never opened as HDF5.\n")
        task_sources.append(dict(path=str(path), sha256=preparer.sha256(path), task_id=task + 5))
        source_rows[str(path)] = [candidate(index, task) for index in range(12)]
        group = {}
        for item in source_rows[str(path)]:
            demo = SyntheticDemo(item["demo"], [])
            demo.datasets["actions"].values = np.zeros((50, 7), dtype=np.float32)
            group[item["demo"]] = demo
        groups[str(path)] = group

    def reservation_admission(_, inputs):
        result = {k: refs[k] for k in refs if k.startswith("fresh_")}
        result["historical_fixture_refs"] = []
        for ref in (refs["fresh_config_ref"], refs["fresh_fixtures_ref"]):
            preparer.checked_json(ref["path"], ref["sha256"], inputs)
        return result, task_sources, [set(), set(), set()], set()

    original_export = preparer.export_actions
    monkeypatch.setattr(preparer, "prior_admission", prior_admission)
    monkeypatch.setattr(preparer, "reservation_admission", reservation_admission)
    monkeypatch.setattr(
        preparer,
        "source_closure",
        lambda _: {**inherited, str(config_path): preparer.sha256(config_path), str(plan): preparer.sha256(plan)},
    )
    monkeypatch.setattr(preparer, "metadata_sources", lambda path, _: copy.deepcopy(source_rows[path]))
    monkeypatch.setattr(
        preparer,
        "export_actions",
        lambda records, inputs: original_export(records, inputs, opener=lambda path, _: SyntheticHandle(groups[path])),
    )
    return dict(config=config_path, source=source, rows=source_rows, output=tmp_path / "outputs" / "native")


def test_full_synthetic_prepare_sorted_json_roundtrip_and_public_guard(preparer, synthetic_preparation):
    path = preparer.prepare(synthetic_preparation["config"])
    prepared = preparer.read_prepared(path, preparer.sha256(path))
    assert len(prepared["records"]) == 30
    assert sum(len(record["anchor_frames"]) for record in prepared["records"]) == 150
    assert not prepared["native_success_verified"] and not prepared["model_training_authorized"]
    assert len(prepared["runtime_source_refs"]) == 1
    runtime_ref = prepared["runtime_source_refs"][0]
    assert prepared["input_sha256"][runtime_ref["path"]] == runtime_ref["sha256"]
    path.write_text(json.dumps(prepared, sort_keys=True))
    assert preparer.read_prepared(path, preparer.sha256(path))["selection"] == prepared["selection"]
    # Reject unexpected read targets before trying to open them, even if a caller
    # supplies the modified prepared SHA instead of the real owner-pinned SHA.
    prepared["input_sha256"]["/must_not_read/fresh_expert.npy"] = "bad"
    path.write_text(json.dumps(prepared, sort_keys=True))
    with pytest.raises(ValueError, match="roster changed or expanded"):
        preparer.read_prepared(path, preparer.sha256(path))


def test_fixed_bad_source_shape_records_false_before_export(preparer, synthetic_preparation, monkeypatch):
    for rows in synthetic_preparation["rows"].values():
        for row in rows:
            row["actions_shape"] = [42, 7]
    monkeypatch.setattr(
        preparer, "export_actions", lambda *a: pytest.fail("no selected actions read before shape admission")
    )
    with pytest.raises(ValueError, match="replacement"):
        preparer.prepare(synthetic_preparation["config"])
    rejection = json.loads((synthetic_preparation["output"] / "preparation" / "selection_rejected.json").read_text())
    assert rejection["metadata_admission"] is False and rejection["selected_action_contents_read"] == 0
    assert len(rejection["selection"]["tasks"]) == 3
    assert all(len(task["selected_demo_ids"]) == 10 for task in rejection["selection"]["tasks"])
    assert not (synthetic_preparation["output"] / "preparation" / "prepared.json").exists()
