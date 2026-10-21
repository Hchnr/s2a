"""026 owner admission, cached-input scope and durable release observations."""

import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_wam_flow_readout")


def test_gpu_release_observations_survive_delay_and_terminal_timeout(owner, monkeypatch, tmp_path):
    success = tmp_path / "success"
    success.mkdir()
    records = iter(
        [
            dict(memory_mib=500, compute_apps=["owned context"], physical_gpu=6),
            dict(memory_mib=0, compute_apps=[], physical_gpu=6),
        ]
    )
    monkeypatch.setattr(owner, "resource_observation", lambda: next(records))
    monkeypatch.setattr(owner.time, "sleep", lambda value: None)
    ref = owner.wait_release(success)
    assert owner.read(ref["path"], ref["sha256"])["released"] is True
    lines = [json.loads(s) for s in (success / "resource_observations.jsonl").read_text().splitlines()]
    assert [record["memory_mib"] for record in lines] == [500, 0]
    failure = tmp_path / "failure"
    failure.mkdir()
    monkeypatch.setattr(owner, "resource_observation", lambda: dict(memory_mib=100, compute_apps=[]))
    with pytest.raises(TimeoutError):
        owner.wait_release(failure, timeout=0)
    assert json.loads((failure / "resource_observations.jsonl").read_text())["memory_mib"] == 100
    assert not (failure / "gpu_after.json").exists()


def test_fit_gate_rejects_producer_or_independent_failure(owner, tmp_path):
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    fit_path = tmp_path / "fit.json"
    fit_path.write_text("{}")
    fit = dict(
        complete=True,
        passed=True,
        prepared_ref=owner.reference(prepared),
        producer_fit_groups=2,
        fit_attempts=2,
        fitted_rows=2880,
        validation_rows=2880,
        eligible_for_sampling=True,
        teacher_gate=dict(passed=True),
        models={},
    )
    audit = dict(
        fit_verified=True,
        eligible_for_sampling=True,
        teacher_gate=dict(passed=True),
        prepared=owner.reference(prepared),
        input_sha256={str(fit_path.resolve()): owner.sha256(fit_path)},
    )
    owner.sampling_admission(fit, audit, prepared, fit_path)
    bad = copy.deepcopy(fit)
    bad["eligible_for_sampling"] = False
    with pytest.raises(ValueError, match="producer teacher gate"):
        owner.sampling_admission(bad, audit, prepared, fit_path)
    bad_audit = copy.deepcopy(audit)
    bad_audit["teacher_gate"]["passed"] = False
    with pytest.raises(ValueError, match="independent teacher gate"):
        owner.sampling_admission(fit, bad_audit, prepared, fit_path)


def test_cached_scope_guard_never_opens_inherited_episode_or_png(owner, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.write_text("cached teacher")
    source = dict(head="synthetic", tracked_source_clean=True)
    prepared = dict(
        schema_version="s2a.flow-readout-preparation.v1",
        config=dict(output_dir=str(tmp_path), cosmos_repo="synthetic"),
        output_root=str(tmp_path),
        external_cosmos=source,
        weight_files={},
        anchors=[dict(arrays_path="must-not-open-episode.npz", image_path="must-not-open.png")],
    )
    for key in ("input_sha256", "source_sha256", "snapshot_sha256"):
        prepared[key] = {str(cache): owner.sha256(cache)}
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(prepared))
    monkeypatch.setattr(owner, "weight_stats", lambda files: None)
    monkeypatch.setattr(owner, "external_identity", lambda repo: source)
    assert owner.read_prepared(path, owner.sha256(path))["anchors"] == prepared["anchors"]
    cache.write_text("changed")
    with pytest.raises(ValueError, match="source/cache SHA"):
        owner.read_prepared(path, owner.sha256(path))


def test_owner_environment_rejected_before_any_input_or_gpu_work(owner, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(ValueError, match="empty CUDA/LD"):
        owner.execute(SimpleNamespace())
