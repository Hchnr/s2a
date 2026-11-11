"""Check independent boundary evidence, scoring, and registered resource limits."""

import subprocess
import sys

import numpy as np
import pytest

from s2a.geometry import pose_error
from scripts.simulation.runtime import prepare_bounded_release_bridge as binding
from scripts.simulation.runtime import run_bounded_release_bridge_feedback as runner


def reference_data():
    old = np.tile(np.eye(4), (40, 1, 1))
    latest = old.copy()
    latest[:, 0, 3] = 0.02
    return (
        {"targets": old, "openings": np.full(40, 0.2)},
        {"targets": latest, "openings": np.full(40, 0.8)},
    )


def test_checker_recomputes_boundary_measurements_and_reference_geometry(monkeypatch):
    old, latest = reference_data()
    decision = {
        "step": 376,
        "absolute_first_endpoint": 377,
        "latest_source_step": 360,
        "source": "latest",
        "reason": "not_closing",
        "poses": latest["targets"][16:],
        "openings": latest["openings"][16:],
    }
    selection = {
        "previous_inner_margin_rad": 1.0,
        "current_joint_margin_rad": 1.0,
        "measured_tcp": np.eye(4),
        "measured_opening": 0.25,
        "decision": decision,
    }
    case = {"controller": {"source_from_site": np.eye(4)}, "tool": np.eye(4)}
    period = {"actual_inner": [{"q": np.zeros(7)} for _ in range(25)]}
    monkeypatch.setattr(runner.hold, "require_checker_independence", lambda: None)
    monkeypatch.setattr(
        runner, "current_joint_site_state", lambda _env: (np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), None)
    )
    monkeypatch.setattr(runner, "current_robot_dynamics", lambda _env, _jacobian: (None, None, np.eye(4)))
    monkeypatch.setattr(runner.base, "measured_gripper", lambda _env: {"opening": 0.25})
    outcome = runner.verify_checker_boundary(None, case, old, latest, selection, period)
    assert outcome["geometry_independently_verified"] and not outcome["selection_recomputed"]
    selection["measured_opening"] = 0.4
    with pytest.raises(AssertionError):
        runner.verify_checker_boundary(None, case, old, latest, selection, period)


def test_bridge_scoring_keeps_selected_and_latest_references_distinct(monkeypatch):
    old, latest = reference_data()
    decision = {"poses": old["targets"][32:].copy(), "openings": np.full(8, 0.4)}
    case = {
        "controller": {"source_from_site": np.eye(4)},
        "tool": np.eye(4),
        "references": {"latest": {"poses": latest["targets"], "cursor": 0}},
    }
    period = {"tcp_error": pose_error(decision["poses"][0], np.eye(4))}
    monkeypatch.setattr(runner, "current_joint_site_state", lambda _env: (None, None, None))
    monkeypatch.setattr(runner, "current_robot_dynamics", lambda _env, _jacobian: (None, None, np.eye(4)))
    monkeypatch.setattr(
        runner.base,
        "score_completed",
        lambda *args, **kwargs: {
            "actual_tcp": np.eye(4),
            "gripper": {"opening": 0.3},
            "reference_errors": {"latest": args[5]},
        },
    )
    scores = runner.score_bridge(None, case, old, decision, period, 0)
    np.testing.assert_allclose(scores["selected_reference_error"], 0)
    assert np.linalg.norm(scores["reference_errors"]["latest"][:3]) == pytest.approx(0.02)
    assert scores["selected_opening_error"] == pytest.approx(-0.1)
    period["tcp_error"] = np.ones(6)
    with pytest.raises(AssertionError):
        runner.score_bridge(None, case, old, decision, period, 0)


def test_24_period_budget_keeps_native_step_count_and_limits():
    config = binding.read_config()[1]
    runtime = {
        "control_steps_completed": 384,
        "counters": {"mj_step": {"completed": 9600}, "model_construct": {"completed": 3}, "context": {"completed": 3}},
        "newly_selected_commands": 24,
        "refiner_calls": {"attempted": 24},
        "refiner_response_evaluations": 0,
        "quality_refiner": {"quality_calls_attempted": 24, "quality_response_evaluations": 1000},
        "gripper_proposals": {"planner_calls": 24, "candidate_sequences": 24 * 6561},
    }
    branch = {
        "complete": True,
        "passed": True,
        "control_qualified": False,
        "runtime": runtime,
        "gripper_history": {"planner_calls": 360, "candidate_sequences": 360 * 6561},
    }
    assert runner.hold.summarize_phase([branch], "producer", config)["native_steps"] == 9600
    runtime["control_steps_completed"] += 1
    with pytest.raises(ValueError, match="control_steps budget"):
        runner.hold.summarize_phase([branch], "producer", config)


def test_checker_import_does_not_load_any_candidate_policy():
    script = "from scripts.simulation.runtime import run_bounded_release_bridge_feedback as r; r.hold.require_checker_independence(); print('independent')"
    process = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=30)
    assert process.stdout.strip() == "independent"
