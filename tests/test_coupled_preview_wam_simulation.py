"""Preserve the qualified control loop while changing reference acquisition."""

import ast
from pathlib import Path

import pytest

from s2a import coupled_preview_wam_simulation as wam

ROOT = Path(__file__).resolve().parents[1]


def test_per_period_control_matches_frozen_oracle_except_reference_query():
    bodies = []
    for filename in ("coupled_pose_preview_control_simulation.py", "coupled_preview_wam_simulation.py"):
        tree = ast.parse((ROOT / "src/s2a" / filename).read_text())
        loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "step"
        ]
        assert len(loops) == 1
        body = [
            node
            for node in loops[0].body
            if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "need_query")
        ]
        bodies.append(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False))
    assert bodies[0] == bodies[1]


def test_missing_physical_gate_stops_before_environment_or_network(monkeypatch, tmp_path):
    def stop(_):
        raise ValueError("physical qualification missing")

    def forbidden(*args, **kwargs):
        pytest.fail("network or simulator reached before control qualification")

    monkeypatch.setattr(wam, "admit_preview_control", stop)
    monkeypatch.setattr(wam, "make_environment", forbidden)
    monkeypatch.setattr(wam.requests, "Session", forbidden)
    with pytest.raises(ValueError, match="physical qualification missing"):
        wam.run_preview_wam_case(
            dict(preview_control_closure_ref={}),
            tmp_path / "must_not_exist",
            {},
            source="wam",
            variant_name="position_hold",
            sampling_seed=42,
            checkpoint="8400_masked",
            config_path=None,
            oracle_reference=None,
            endpoint_observer=lambda *a: None,
            response_parameters=None,
        )
    assert not (tmp_path / "must_not_exist").exists()


def test_wam_entry_rejects_expert_reference_before_admission(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("expert-reference rejection must precede admission")

    monkeypatch.setattr(wam, "admit_preview_control", forbidden)
    with pytest.raises(ValueError, match="no expert reference"):
        wam.run_preview_wam_case(
            {},
            tmp_path,
            {},
            source="wam",
            variant_name="position_hold",
            sampling_seed=42,
            config_path=None,
            oracle_reference=([1], [1], {}),
            endpoint_observer=lambda *a: None,
            response_parameters=None,
        )
