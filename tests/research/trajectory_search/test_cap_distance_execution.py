"""Fixed scientific budgets and owned-process timeout/source admission."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.tracking.trajectory_search.execute_cap_distance_gn as owner
from s2a.infrastructure.publication import publish_result
from scripts.tracking.trajectory_search.check_cap_distance_gn import validate_spec


@pytest.fixture
def spec():
    root = Path(__file__).resolve().parents[3] / "configs/tracking/trajectory_search"
    return (
        json.loads((root / "cap_distance_gn_20261102.json").read_bytes()),
        json.loads((root / "state_vector_dual_recovery_20261029.json").read_bytes()),
    )


def test_exact_configuration_is_admitted(spec):
    configured, original = spec
    assert validate_spec(configured, original) == original


@pytest.mark.parametrize("fault", ["budget", "gate", "arithmetic", "timeout"])
def test_changed_scientific_gate_budget_or_arithmetic_is_rejected(spec, fault):
    configured, original = spec
    if fault == "budget":
        configured["producer_response_budget"] += 1
    elif fault == "gate":
        configured["search_config"]["minimum_actual_reduction"] /= 10
    elif fault == "arithmetic":
        configured["cap_loss_arithmetic"] = "direct_sqrt"
    else:
        configured["owner_timeout_seconds"] += 1
    with pytest.raises(ValueError):
        validate_spec(configured, original)


@pytest.mark.parametrize("mode", ["normal", "bad_qa", "timeout", "source_changed", "exit_race", "hard_timeout"])
def test_owner_admission_timeout_and_source_checks(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(owner, "OWNER", tmp_path / "owner")
    monkeypatch.setattr(owner, "OUTPUT", tmp_path / "experiment")
    sources = dict(sources={"algorithm.py": "original"}, current_map_ref={}, driver_ref={}, head="before")
    after_ref = publish_result(tmp_path / "qa_sources.json", sources)
    qa_ref = publish_result(
        tmp_path / "qa.json",
        dict(passed=True, all_tests_passed=True, source_unchanged=True, skipped_count=0, source_after_ref=after_ref),
    )
    calls, signals, snapshots = [], [], []

    def snapshot(root):
        snapshots.append(root)
        result = dict(sources)
        if len(snapshots) > 1:
            result["head"] = "after-normal-commit"
            if mode == "source_changed":
                result["sources"] = {"algorithm.py": "changed"}
        return result

    monkeypatch.setattr(owner, "snapshot", snapshot)

    def signal_group(pid, sig):
        signals.append((pid, sig))
        if mode == "exit_race":
            raise ProcessLookupError("Child already exited")

    monkeypatch.setattr(owner.os, "killpg", signal_group)

    class Process:
        pid = 12345
        returncode = None

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            assert kwargs["start_new_session"] is True
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            publish_result(owner.OUTPUT / "result.json", dict(complete=True, passed=True, sources_unchanged=True))
            self.waits = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if (mode in ("timeout", "exit_race") and len(self.waits) == 1) or (
                mode == "hard_timeout" and len(self.waits) <= 2
            ):
                raise owner.subprocess.TimeoutExpired(calls[0][0], timeout)
            self.returncode = -15 if mode == "timeout" else 0
            return self.returncode

    monkeypatch.setattr(owner.subprocess, "Popen", Process)
    sha = qa_ref["sha256"] if mode != "bad_qa" else hashlib.sha256(b"wrong").hexdigest()
    code = owner.main(SimpleNamespace(qa_result=qa_ref["path"], qa_sha=sha))
    result = json.loads((owner.OWNER / "result.json").read_bytes())
    assert (code == 0) is (mode == "normal")
    assert result["passed"] is (mode == "normal")
    if mode == "bad_qa":
        assert not calls and not result["launched"]
    else:
        assert len(calls) == 1 and result["actual_wait"] and result["reaped"]
    if mode in ("timeout", "exit_race", "hard_timeout"):
        assert result["timed_out"] and result["signal_sent"]
        assert signals and all(pid == 12345 for pid, _ in signals)
    else:
        assert not signals
