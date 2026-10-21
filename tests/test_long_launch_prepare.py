"""A passed score/count summary cannot bypass the explicit release gate."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_prepare_rejects_false_normal_release_before_creating_any_gate(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("long_prepare_test", root / "scripts/prepare_wam_long_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repository = tmp_path / "repo"
    outputs = repository / "outputs"
    outputs.mkdir(parents=True)
    qualification = outputs / "qualification.json"
    qualification.write_text(
        json.dumps(
            dict(
                qualification_passed=True,
                all_cases_complete=True,
                native_trials=3,
                oracle_trials=3,
                physical_steps=7200,
                normal_process_release=False,
                input_sha256={},
            )
        )
    )
    prior = outputs / "prior.json"
    prior.write_text(json.dumps(dict(prior_scopes_passed=True, input_sha256={})))
    monkeypatch.setattr(module, "REPO", repository)
    monkeypatch.setattr(module, "PRIOR", prior)
    monkeypatch.setattr(module, "PRIOR_SHA", hashlib.sha256(prior.read_bytes()).hexdigest())
    target = outputs / "never_created_launch_gates"
    with pytest.raises(ValueError, match="all six actual1200-step"):
        module.prepare(qualification, hashlib.sha256(qualification.read_bytes()).hexdigest(), target)
    assert not target.exists()
