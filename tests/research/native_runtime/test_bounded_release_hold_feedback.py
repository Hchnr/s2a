"""The physical hold must use completed state and commit only actual execution."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.wam.tracking.preview.bounded_release_revision import BoundedReleaseRevision, ReleaseRevisionConfig
from scripts.simulation.runtime import prepare_bounded_release_hold as binding
from scripts.simulation.runtime import prepare_causal_event_bridge_feedback as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_bounded_release_hold_feedback as runner


def fixture(closing=True):
    config = binding.read_config()[1]
    selector = BoundedReleaseRevision(ReleaseRevisionConfig(8, 8, 0.04, 0.04, 0.03, 0.15, 0.05))
    old_pose, new_pose = np.tile(np.eye(4), (40, 1, 1)), np.tile(np.eye(4), (40, 1, 1))
    new_pose[:, 0, 3] = 0.01
    old_open, new_open = np.full(40, 0.2), np.full(40, 0.9)
    old = {
        "targets": old_pose,
        "openings": old_open,
        "fixed_targets": old_pose[16:32],
        "fixed_openings": old_open[16:32],
    }
    latest = {"targets": new_pose, "openings": new_open, "anchor": np.eye(4)}
    grip = SimpleNamespace(state=lambda: {"staged": False, "faulted": False, "latch": {"closing": closing}})
    policy = SimpleNamespace(last_inner_loss=False, gripper=grip)
    periods = [
        {"step": 360 + i, "actual_inner": [{"q": np.zeros(7)} for _ in range(25)], "minimum_inner_margin_rad": 1.0}
        for i in range(8)
    ]
    receipt = {
        "continuation_partial": {"periods": periods},
        "control_steps_attempted": 368,
        "control_steps_completed": 368,
    }
    limits = np.tile([-1.0, 1.0], (7, 1))
    return config, selector, old, latest, policy, receipt, limits


def start(monkeypatch, *, closing=True):
    config, selector, old, latest, policy, receipt, limits = fixture(closing)
    switches = []
    monkeypatch.setattr(runner.switch, "switch_reference", lambda p, selected, r: switches.append(selected))
    stop = runner.start_hold(
        policy,
        selector,
        old,
        latest,
        config,
        receipt,
        measured_tcp=np.eye(4),
        measured_opening=0.25,
        limits=limits,
    )
    return stop, selector, receipt, switches


def test_boundary_rebinds_only_completed_history_and_includes_the_future_bridge(monkeypatch):
    stop, selector, receipt, switches = start(monkeypatch)
    assert stop is None and selector.state()["staged"] and len(switches) == 1
    assert receipt["release_selection"]["decision"]["reason"] == "bounded_hold"
    selected = switches[0]
    np.testing.assert_array_equal(selected["openings"][:16], np.full(16, 0.2))
    assert selected["openings"][16] == pytest.approx(0.2 * 7 / 8 + 0.9 / 8)
    assert selected["openings"][23] == pytest.approx(0.9)
    assert not receipt["release_selection"]["bridge_executed"]


def test_saved_completed_physical_boundary_reconstructs_the_audited_selection(monkeypatch):
    inputs = binding.load_inputs()
    prior = prefix.read(inputs["prior_long_refs"]["producer"])["runtime"]
    periods = prior["continuation"]["periods"][:8]
    scores = periods[-1]["completed_feedback"]["scores"]
    old, latest = previous.selected_references(inputs["case"])
    grip = SimpleNamespace(state=lambda: {"staged": False, "faulted": False, "latch": {"closing": True}})
    policy = SimpleNamespace(last_inner_loss=False, gripper=grip)
    selector = BoundedReleaseRevision(ReleaseRevisionConfig(8, 8, 0.04, 0.04, 0.03, 0.15, 0.05))
    receipt = {
        "continuation_partial": {"periods": periods},
        "control_steps_attempted": 368,
        "control_steps_completed": 368,
    }
    selected = []
    monkeypatch.setattr(runner.switch, "switch_reference", lambda p, ref, r: selected.append(ref))
    stop = runner.start_hold(
        policy,
        selector,
        old,
        latest,
        inputs["config"],
        receipt,
        measured_tcp=scores["actual_tcp"],
        measured_opening=scores["gripper"]["opening"],
        limits=prior["final_prefix_inner"]["limits"],
    )
    shadow = prefix.read(prefix.read(inputs["prior_shadow_refs"]["producer"])["studied_ref"])
    actual = receipt["release_selection"]["decision"]
    assert stop is None and actual["reason"] == shadow["decision"]["reason"] == "bounded_hold"
    np.testing.assert_allclose(actual["poses"][:16], shadow["decision"]["poses"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(actual["openings"][:16], shadow["decision"]["openings"], atol=1e-10, rtol=0)
    assert len(selected) == 1 and selector.state()["staged"]


def test_guard_not_eligible_stops_before_dispatch_without_committing(monkeypatch):
    stop, selector, receipt, switches = start(monkeypatch, closing=False)
    assert stop["stop"]["reason"] == "release_guard_not_eligible" and not switches
    assert receipt["release_selection"]["decision"]["reason"] == "not_closing"
    assert not selector.state()["staged"] and selector.state()["mode"] == "idle"


def test_observed_inner_buffer_loss_cannot_be_converted_into_a_hold(monkeypatch):
    config, selector, old, latest, policy, receipt, limits = fixture()
    for sample in receipt["continuation_partial"]["periods"][-1]["actual_inner"]:
        sample["q"][0] = 0.98
    receipt["continuation_partial"]["periods"][-1]["minimum_inner_margin_rad"] = 0.02
    monkeypatch.setattr(runner.switch, "switch_reference", lambda *args: pytest.fail("unsafe rebind"))
    with pytest.raises(ValueError, match="buffer loss"):
        runner.start_hold(
            policy, selector, old, latest, config, receipt, measured_tcp=np.eye(4), measured_opening=0.25, limits=limits
        )
    assert not selector.state()["staged"]


@pytest.mark.parametrize("completed", [0, 3, 8])
def test_reference_state_requires_actual_completed_periods(monkeypatch, completed):
    stop, selector, receipt, _ = start(monkeypatch)
    assert stop is None
    receipt["continuation_partial"]["periods"].extend({} for _ in range(completed))
    receipt["control_steps_attempted"] += completed
    receipt["control_steps_completed"] += completed
    runner.finish_selection(selector, receipt)
    state = selector.state()
    assert receipt["release_selection"]["completed_hold_steps"] == completed
    assert not state["staged"]
    assert state["mode"] == ("held" if completed == 8 else "idle")
    assert state["faulted"] is (0 < completed < 8)


def test_uncertain_partial_physical_period_remains_staged_and_requires_reconstruction(monkeypatch):
    _, selector, receipt, _ = start(monkeypatch)
    receipt["control_steps_attempted"] += 1
    runner.finish_selection(selector, receipt)
    assert selector.state()["staged"]
    assert receipt["release_selection"]["reconstruction_required"]
    assert receipt["release_selection"]["completed_hold_steps"] == 0


def test_checker_records_a_nonrecomputed_guard_stop():
    expected = {"reason": "release_guard_not_eligible", "at_step": 368, "next_action_dispatched": False}
    outcome = {"stop": None, "periods": [{} for _ in range(8)]}
    actual = runner.reconcile_checker(outcome, expected, current_margin=0.5, prefix_inner_loss=False)
    assert actual["stop"] == expected and actual["decision_stop_recorded_not_recomputed"]


def test_budget_summary_uses_registered_model_and_gripper_limits():
    config = binding.read_config()[1]
    runtime = {
        "control_steps_completed": 376,
        "counters": {"mj_step": {"completed": 9400}, "model_construct": {"completed": 3}, "context": {"completed": 3}},
        "newly_selected_commands": 16,
        "refiner_calls": {"attempted": 16},
        "refiner_response_evaluations": 0,
        "quality_refiner": {"quality_calls_attempted": 16, "quality_response_evaluations": 1000},
        "gripper_proposals": {"planner_calls": 8, "candidate_sequences": 52488},
    }
    branch = {
        "complete": True,
        "passed": True,
        "control_qualified": False,
        "runtime": runtime,
        "gripper_history": {"planner_calls": 360, "candidate_sequences": 2361960},
    }
    result = runner.summarize_phase([branch], "producer", config)
    assert result["model_initializations"] == 3 and result["gripper_planner_calls"] == 368
    runtime["counters"]["model_construct"]["completed"] = 4
    with pytest.raises(ValueError, match="model_initializations budget"):
        runner.summarize_phase([branch], "producer", config)


def test_checker_starts_without_candidate_modules_in_a_fresh_process():
    script = "from scripts.simulation.runtime import run_bounded_release_hold_feedback as r; r.require_checker_independence(); print('independent')"
    completed = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=30)
    assert completed.stdout.strip() == "independent"
