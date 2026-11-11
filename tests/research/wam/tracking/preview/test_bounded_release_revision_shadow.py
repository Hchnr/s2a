"""Check shadow causality, source alignment and independent reconstruction."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports import check_bounded_release_revision as checker
from scripts.wam.tracking.preview import bind_bounded_release_shadow as binding
from scripts.wam.tracking.preview import replay_bounded_release_revision as producer


def config():
    return prefix.read(prefix.reference(binding.CONFIG))


def synthetic_baseline():
    poses = np.tile(np.eye(4), (40, 1, 1))
    chunks = [
        {"step": 0, "effective_targets_tcp": poses.copy(), "openings": np.full(40, 0.2)},
        {"step": 8, "effective_targets_tcp": poses.copy(), "openings": np.full(40, 0.9)},
        {"step": 16, "effective_targets_tcp": poses.copy(), "openings": np.full(40, 0.9)},
    ]
    rows = [
        {
            "measured_source": np.eye(4),
            "opening_before": 0.25,
            "gripper": {"latch": {"closing_episode": index == 7}},
        }
        for index in range(24)
    ]
    return chunks, rows, np.eye(4)


def test_shadow_records_conflict_without_fabricating_safety_or_an_intervention():
    item = {"case_id": "synthetic"}
    chunks, rows, tool = synthetic_baseline()
    result = producer.shadow_case(item, chunks, rows, tool, config())
    assert result["summary"]["conflicts_without_safety"] == 1
    assert result["summary"]["unknown_inner_conflicts"] == 1
    assert result["summary"]["holds"] == result["summary"]["bridges"] == 0
    assert result["records"][1]["reason"] == "unknown_inner_margin"
    assert result["records"][1]["previous_inner_margin_rad"] is None
    checked = checker.verify_case(item, {"case_id": "synthetic", **result}, chunks, rows, tool, config())
    assert checked == result["summary"]


def test_independent_shadow_check_rejects_a_tampered_decision():
    item = {"case_id": "synthetic"}
    chunks, rows, tool = synthetic_baseline()
    result = {"case_id": "synthetic", **producer.shadow_case(item, chunks, rows, tool, config())}
    changed = copy.deepcopy(result)
    changed["records"][1]["reason"] = "bounded_hold"
    with pytest.raises(ValueError, match="Independent baseline decision"):
        checker.verify_case(item, changed, chunks, rows, tool, config())


def test_checker_import_is_independent_in_a_fresh_process():
    script = (
        "from scripts.verification.reference_ports import check_bounded_release_revision as c; "
        "c.independent(); print('independent')"
    )
    completed = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=30)
    assert completed.stdout.strip() == "independent"


def test_oracle_binding_checks_every_generated_endpoint_against_the_original_reference(tmp_path):
    count = 10
    poses = np.tile(np.eye(4), (count, 1, 1))
    poses[:, 0, 3] = np.arange(count) * 0.001
    openings = np.linspace(0.2, 0.7, count)
    chunks = [
        {
            "step": step,
            "anchor_tcp": np.eye(4).tolist(),
            "effective_targets_tcp": poses[step:].tolist(),
            "openings": openings[step:].tolist(),
        }
        for step in (0, 8)
    ]
    rows = [
        {
            "step": step,
            "chunk_index": step // 8,
            "reference_index": step % 8,
            "measured_source": np.eye(4).tolist(),
            "desired_tcp": poses[step].tolist(),
            "gripper": {"preview": {"original_reference": float(openings[step])}},
        }
        for step in range(count)
    ]
    for name, value in (
        ("chunks.json", chunks),
        ("rows.json", rows),
        ("tool.json", {"source_from_tcp": np.eye(4).tolist()}),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    np.savez(tmp_path / "trajectory.npz", poses=poses, openings=openings)
    item = {
        "family": "correct_reference",
        "steps": count,
        "format": "world_reference",
        "refs": {
            "chunks": prefix.reference(tmp_path / "chunks.json"),
            "rows": prefix.reference(tmp_path / "rows.json"),
            "tool": prefix.reference(tmp_path / "tool.json"),
            "trajectory": prefix.reference(tmp_path / "trajectory.npz"),
        },
    }
    bound, saved_rows, _ = binding.read_case(item)
    assert len(bound) == 2 and len(saved_rows) == count
    chunks[0]["openings"][9] = 0.01
    (tmp_path / "chunks.json").write_text(json.dumps(chunks))
    item["refs"]["chunks"] = prefix.reference(tmp_path / "chunks.json")
    with pytest.raises(AssertionError):
        binding.read_case(item)


def test_recorded_completed_state_has_real_inner_evidence_and_auditable_hold():
    inputs = binding.load_inputs()
    produced = producer.studied_case(inputs, config())
    summary = checker.verify_studied(inputs, produced, config())
    assert summary["step"] == 368 and summary["reason"] == "bounded_hold"
    assert summary["previous_inner_margin_rad"] >= config()["joint_buffer_rad"]
    assert produced["selection_executed"] is False
