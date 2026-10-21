"""027 ownership boundaries do not grant GPU or repeat-fit permission."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def module():
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("scaled_owner_test", REPO / "scripts/run_wam_scaled_readout.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def decisions(passed):
    gate = dict(passed=passed, checks=dict(fixed_teacher_gate=passed))
    producer = dict(
        candidate_fits=1,
        fit_attempts=1,
        producer_fit_groups=1,
        reused_controls=1,
        fitted_rows=2880,
        validation_rows=2880,
        teacher_gate=gate,
        eligible_for_runtime_qualification=passed,
        eligible_for_sampling=False,
    )
    audit = dict(
        independent_candidate_reconstructions=1,
        fit_verified=True,
        teacher_gate=gate,
        eligible_for_runtime_qualification=passed,
        eligible_for_sampling=False,
    )
    return producer, audit


@pytest.mark.parametrize("passed", [False, True])
def test_teacher_decision_never_authorizes_sampling(passed):
    owner = module()
    producer, audit = decisions(passed)
    owner.fit_admission(producer, audit)
    for side in (producer, audit):
        changed = copy.deepcopy(side)
        changed["eligible_for_sampling"] = True
        with pytest.raises(ValueError, match="cannot authorize"):
            owner.fit_admission(changed if side is producer else producer, changed if side is audit else audit)


def test_mismatched_gate_or_repeated_fit_rejected():
    owner = module()
    producer, audit = decisions(False)
    audit["eligible_for_runtime_qualification"] = True
    with pytest.raises(ValueError, match="decisions differ"):
        owner.fit_admission(producer, audit)
    producer, audit = decisions(False)
    producer["candidate_fits"] = 2
    with pytest.raises(ValueError, match="single-fit"):
        owner.fit_admission(producer, audit)


def test_planned_children_are_cpu_and_independent(tmp_path):
    owner = module()
    prepared = dict(output_root=str(tmp_path), config=dict(cosmos_repo="/readonly/reference"))
    args = SimpleNamespace(prepared=tmp_path / "prepared.json", prepared_sha256="a" * 64)
    worker, audit = owner.planned_commands(args, prepared)
    assert worker[:2] == [str(REPO / ".venv/bin/python"), "scripts/fit_wam_scaled_readout.py"]
    assert audit[:2] == [str(REPO / ".venv/bin/python"), "scripts/verify_wam_scaled_readout.py"]
    env = owner.environment(prepared, False)
    assert env["CUDA_VISIBLE_DEVICES"] == env["LD_LIBRARY_PATH"] == ""
    assert not any(word in " ".join(worker + audit) for word in ("sampling.py", "probe_", "--gpu"))


def test_wrong_preparation_digest_rejected_before_children(tmp_path, monkeypatch):
    owner = module()
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(dict(schema_version="s2a.scaled-readout-preparation.v1")))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(
        owner, "owned_command", lambda *a, **k: pytest.fail("child started before preparation admission")
    )
    args = SimpleNamespace(prepared=path, prepared_sha256="0" * 64)
    with pytest.raises(ValueError, match="artifact SHA differs"):
        owner.execute(args)


def test_existing_output_preserved_before_launch(tmp_path, monkeypatch):
    owner = module()
    fit = tmp_path / "fit"
    fit.mkdir()
    original = fit / "original.txt"
    original.write_text("preserve")
    prepared = dict(output_root=str(tmp_path))
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("{}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(owner, "read_prepared", lambda *args: prepared)
    peer = dict(
        passed=True,
        phase="prepare",
        preparation_verified=True,
        prepared=owner.reference(prepared_path),
        input_sha256={},
    )
    monkeypatch.setattr(owner, "read", lambda *args: peer)
    monkeypatch.setattr(owner, "owned_command", lambda *a, **k: pytest.fail("existing output was reused"))
    args = SimpleNamespace(
        prepared=prepared_path,
        prepared_sha256="unused",
        source_peer=tmp_path / "peer.json",
        source_peer_sha256="unused",
        dry_run=False,
    )
    with pytest.raises(ValueError, match="refuses existing"):
        owner.execute(args)
    assert original.read_text() == "preserve"
    assert not (tmp_path / "orchestration").exists()
