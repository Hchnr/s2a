"""041 root-only metadata and source/command contracts; no real labels or jobs."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from s2a.artifacts import REPO


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    value = importlib.import_module("prepare_wam_clean_training_continuation")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("preparation must not decode future archives"))
    return value


def original(root):
    return dict(
        root=str(root / "outputs/wam_stability_20261021/clean_residual_training_20261023"),
        execution_authorized=False,
        readiness=dict(ready_for_execution=False, reason="metadata only"),
        training_requests=[dict(sampling_seed=123, request=dict(seed=123, prompt="unchanged task"))],
        input_sha256={"/opaque/future.npz": "a" * 64},
        config=dict(ridge=0.1, feature_terms=[]),
    )


def test_derived_preparation_changes_only_root_and_never_mutates_original(module):
    before = original(REPO)
    frozen = copy.deepcopy(before)
    derived = module.derive_prepared(before)
    assert before == frozen
    assert derived == dict(before, root=str(REPO / module.NEW_ROOT_REL))
    derived["training_requests"][0]["request"]["seed"] = 999
    assert before == frozen
    assert before["input_sha256"] == derived["input_sha256"]


@pytest.mark.parametrize("fault", ["root", "authorized", "ready"])
def test_derivation_rejects_rebound_or_promoted_original(module, fault):
    before = original(REPO)
    if fault == "root":
        before["root"] += "/unregistered"
    elif fault == "authorized":
        before["execution_authorized"] = True
    else:
        before["readiness"]["ready_for_execution"] = True
    with pytest.raises(ValueError, match="unchanged original"):
        module.derive_prepared(before)


@pytest.fixture
def repository(module, monkeypatch, tmp_path):
    root = tmp_path / "isolated"
    root.mkdir()
    monkeypatch.setattr(module, "REPO", root)
    for name in ("src/s2a/core.py", "scripts/entry.py", "tests/test_contract.py", *module.EXTRA_SOURCES):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic fixture\n")
    config = json.loads((REPO / "configs/wam_clean_training_continuation_20261023.json").read_text())
    for name in config["required_test_paths"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic test path fixture\n")
    return root, config


@pytest.mark.parametrize(
    "fault",
    [None, "root", "retry", "fit_old", "fresh", "total_denoise", "old_lower", "init", "timeout", "GPU", "test"],
)
def test_fixed_one_new_attempt_and_aggregate_consumption(module, repository, fault):
    root, config = repository
    if fault == "root":
        config["output_dir"] = "outputs/old-attempt"
    elif fault == "retry":
        config["automatic_retry"] = True
    elif fault == "fit_old":
        config["old_data_used_for_fit"] = True
    elif fault == "fresh":
        config["fresh_performance_authorized"] = True
    elif fault == "total_denoise":
        config["aggregate_budget"]["total_denoise_upper_bound"] = 55965
    elif fault == "old_lower":
        config["aggregate_budget"]["original_denoise_lower_bound"] = 700
    elif fault == "init":
        config["aggregate_budget"]["total_model_initializations_upper_bound"] = 4
    elif fault == "timeout":
        config["stage_timeouts_s"]["D0"] = 86400
    elif fault == "GPU":
        config["gpu_uuid"] = "0"
    elif fault == "test":
        (root / config["required_test_paths"][-1]).unlink()
    if fault is None:
        module.config_contract(config)
    else:
        with pytest.raises(ValueError):
            module.config_contract(config)


def bundle_fixture(module, root):
    values = {}
    for index in range(11):
        path = root / module.BUNDLE_REL / f"entry_{index}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VALUE = {index}\n")
        values[str(path)] = module.sha(path)
    return dict(source_sha256=values)


def test_source_closure_includes_actual_bundle_and_separately_snapshots_it(module, repository):
    root, _ = repository
    bundle = bundle_fixture(module, root)
    paths = module.source_paths(bundle)
    assert set(bundle["source_sha256"]) <= {str(path) for path in paths}
    assert {root / name for name in module.EXTRA_SOURCES} <= set(paths)
    selected = {str(root / "src/s2a/core.py"): module.sha(root / "src/s2a/core.py"), **bundle["source_sha256"]}
    execution = root / module.NEW_ROOT_REL / "execution"
    snapshots, refs = module.freeze_sources(execution, selected)
    assert len(snapshots) == len(refs) == 12
    assert refs[str(root / "src/s2a/core.py")]["path"] == str(execution / "source/repo/src/s2a/core.py")
    for path in bundle["source_sha256"]:
        target = Path(refs[path]["path"])
        assert target.parent == execution / "source/bundle"
        assert target.read_bytes() == Path(path).read_bytes()
    with pytest.raises(FileExistsError):
        module.freeze_sources(execution, selected)


def test_snapshot_rejects_changed_source_before_any_copy(module, repository):
    root, _ = repository
    path = root / "src/s2a/core.py"
    expected = module.sha(path)
    path.write_text("# changed\n")
    with pytest.raises(ValueError, match="changed before"):
        module.freeze_sources(root / module.NEW_ROOT_REL / "execution", {str(path): expected})


def test_arbitrary_outputs_file_cannot_be_admitted_as_executable_source(module, repository):
    root, _ = repository
    path = root / "outputs/unqualified.py"
    path.parent.mkdir()
    path.write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="only qualified bundle"):
        module.freeze_sources(root / module.NEW_ROOT_REL / "execution", {str(path): module.sha(path)})


def test_source_roster_refuses_bundle_omission(module, repository):
    root, _ = repository
    bundle = bundle_fixture(module, root)
    bundle["source_sha256"].pop(next(iter(bundle["source_sha256"])))
    with pytest.raises(ValueError, match="complete main"):
        module.source_paths(bundle)


def test_static_stage_commands_have_no_self_sha_or_fresh_and_bind_real_entrypoints(module):
    contract = module.supervisor_contract()
    assert contract["stage_order"] == ["D0", "W1", "D1", "W2", "STATE", "DEVELOPMENT"]
    assert contract["stage_timeouts_s"] == module.TIMEOUTS
    for stage, command in contract["stage_commands"].items():
        assert set(command) == {"entrypoint", "arguments"}
        assert command["entrypoint"].startswith(module.BUNDLE_REL + "/")
        assert command["arguments"] == (["--stage", stage] if stage in ("D0", "W1", "D1", "W2") else [])
        assert contract["result_paths"][stage] == f"orchestration/{stage}/result.json"
    assert "execution-registration-sha256" not in json.dumps(contract)
    assert module.stage_entrypoints()["W1"] == module.stage_entrypoints()["W2"] == "scripts/fit_wam_clean_training.py"


def test_contract_keeps_failed_attempt_as_evidence_and_excludes_its_data(module):
    config = dict(
        original_execution_ref=dict(path="/original", sha256="a" * 64),
        incident_ref=dict(path="/incident", sha256="b" * 64),
    )
    value = module.continuation_contract(config)
    assert value == dict(
        protocol="041", **config, old_data_used_for_fit=False, one_new_complete_attempt=True, automatic_retry=False
    )
