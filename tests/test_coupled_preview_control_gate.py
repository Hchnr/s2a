"""Admission rejects missing closure, forged aggregate success, and changed evidence."""

import json
from pathlib import Path

import pytest

from s2a import coupled_preview_control_gate as gate
from s2a.artifacts import sha256


def make_closure(tmp_path, monkeypatch, *, failing_case=None):
    for name, value in (
        ("RUN", tmp_path / "run"),
        ("OWNER", tmp_path / "owner"),
        ("REPAIR", tmp_path / "repair"),
        ("CONFIG", tmp_path / "config.json"),
    ):
        monkeypatch.setattr(gate, name, value)
    sources = {}

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        sources[str(path)] = sha256(path)
        return dict(path=str(path), sha256=sources[str(path)])

    def completed(directory):
        child = dict(pid=2147483647, start_ticks=1)
        save(directory / "launch.json", dict(child=child))
        save(directory / "exit.json", dict(actual_wait=True, returncode=0, automatic_retry=False))
        save(
            directory / "release.json",
            dict(
                **child,
                actual_wait=True,
                kill_sent=False,
                reaped=True,
                returncode=0,
                signal_sent=False,
                original_error=None,
                only_owned_child_signalled=True,
            ),
        )

    cases, profiles = [], []
    for i in range(18):
        steps = 250 if i < 17 else 657
        tracking = dict(position_rmse_m=0.011 if i == failing_case else 0.001, so3_rmse_rad=0.01)
        outcome = dict(stable=True)
        control = dict(steps=steps, outcome=outcome, tracking=dict(admitted=tracking))
        save(
            gate.RUN / f"case{i:02d}/independent_verified.json",
            dict(complete=True, mechanical_labels_images_passed=True, performance_qualified=True, control=control),
        )
        cases.append(dict(case=i, record=f"r{i}", physics_steps=steps, outcome=outcome, tracking=tracking))
        profiles.append(dict(record=dict(name=f"r{i}"), physics_steps=steps))
        completed(gate.RUN / "orchestration" / f"case{i:02d}")
        completed(gate.REPAIR / "case00" if i == 0 else gate.RUN / "verification" / f"case{i:02d}")
    completed(gate.OWNER / "owner_wait")
    save(
        gate.REPAIR / "verified.json",
        dict(complete=True, reuses_original_collection=True, actual_owned_wait_recorded=True),
    )
    save(gate.CONFIG, dict(profiles=profiles))
    summary = save(
        gate.RUN / "regression_summary.json",
        dict(
            complete=True,
            performance_qualified=True,
            execution_and_independent_checks_passed=True,
            candidate_stable=18,
            candidate_trials=18,
            cases=cases,
            physics_steps=4907,
        ),
    )
    return save(
        gate.RUN / "closure_verified.json",
        dict(
            complete=True,
            owner_normal_wait_verified=True,
            mechanical_qualification=True,
            performance_qualified=True,
            verified_cases=18,
            physics_steps=4907,
            summary_ref=summary,
            source_sha256=dict(sources),
        ),
    )


def test_complete_synthetic_receipts_only_admit_control_not_model(tmp_path, monkeypatch):
    ref = make_closure(tmp_path, monkeypatch)
    result = gate.admit_preview_control(ref)
    assert result["physical_control_qualified"]
    assert not result["wam_execution_authorized"] and not result["fresh_authorized"]


def test_forged_aggregate_cannot_hide_one_failed_case(tmp_path, monkeypatch):
    ref = make_closure(tmp_path, monkeypatch, failing_case=9)
    with pytest.raises(ValueError, match="original performance gate"):
        gate.admit_preview_control(ref)


def test_changed_physical_evidence_fails_digest_check(tmp_path, monkeypatch):
    ref = make_closure(tmp_path, monkeypatch)
    (gate.RUN / "case00/independent_verified.json").write_text("{}")
    with pytest.raises(ValueError):
        gate.admit_preview_control(ref)


def test_prefix_without_closure_cannot_be_admitted(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "RUN", tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.admit_preview_control(dict(path=str(Path(tmp_path) / "closure_verified.json"), sha256="0" * 64))
