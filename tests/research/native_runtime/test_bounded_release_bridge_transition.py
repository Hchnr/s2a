"""Exercise completed-history transitions and partial physical execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.wam.tracking.preview.bounded_release_revision import (
    BoundedReleaseRevision,
    ReferenceChunk,
    ReleaseRevisionConfig,
)
from scripts.simulation.runtime import bounded_release_bridge_transition as transition


def test_horizon_extension_is_atomic_on_stops_and_errors(monkeypatch):
    policy = SimpleNamespace(max_steps=16, cursor=0, poses=np.zeros((40, 4, 4)))
    receipt = {"continuation_partial": {"periods": [None] * 8}}
    monkeypatch.setattr(transition.hold, "start_hold", lambda p, *a, **k: {"stop": {"reason": "guard"}})
    assert transition.start_extended_hold(policy, None, None, None, None, receipt)["stop"]["reason"] == "guard"
    assert policy.max_steps == 16 and not receipt["horizon_extension"]["completed"]

    def fail(p, *args, **kwargs):
        assert p.max_steps == 24
        raise ValueError("failed reference switch")

    monkeypatch.setattr(transition.hold, "start_hold", fail)
    with pytest.raises(ValueError, match="failed reference switch"):
        transition.start_extended_hold(policy, None, None, None, None, receipt)
    assert policy.max_steps == 16
    monkeypatch.setattr(transition.hold, "start_hold", lambda p, *a, **k: None)
    assert transition.start_extended_hold(policy, None, None, None, None, receipt) is None
    assert policy.max_steps == 24 and receipt["horizon_extension"]["completed"]


def fixture(monkeypatch, *, closing=True):
    old_pose = np.tile(np.eye(4), (40, 1, 1))
    new_pose = old_pose.copy()
    new_pose[:, 0, 3] = 0.01
    old_opening, new_opening = np.full(40, 0.2), np.full(40, 0.9)
    old = {"targets": old_pose, "openings": old_opening}
    latest = {"targets": new_pose, "openings": new_opening, "anchor": np.eye(4)}
    selector = BoundedReleaseRevision(ReleaseRevisionConfig(8, 8, 0.04, 0.04, 0.03, 0.15, 0.05))
    first = selector.prepare(
        ReferenceChunk(360, new_pose, new_opening),
        committed=ReferenceChunk(344, old_pose, old_opening),
        step=368,
        measured_tcp=np.eye(4),
        measured_opening=0.25,
        closing_episode=True,
        previous_inner_margin_rad=1.0,
    )
    selector.commit(first, completed_steps=8)
    gripper = SimpleNamespace(state=lambda: {"staged": False, "faulted": False, "latch": {"closing": closing}})
    switches = []
    policy = SimpleNamespace(
        last_inner_loss=False,
        gripper=gripper,
        poses=old_pose.copy(),
        openings=old_opening.copy(),
        rebind_reference=lambda *args, **kwargs: switches.append((args, kwargs)) or {"completed_offset": 16},
    )
    monkeypatch.setattr(transition.switch, "control_history", lambda _policy: {"position": np.arange(7)})
    periods = [
        {"step": 360 + i, "actual_inner": [{"q": np.zeros(7)} for _ in range(25)], "minimum_inner_margin_rad": 1.0}
        for i in range(16)
    ]
    receipt = {
        "continuation_partial": {"periods": periods},
        "control_steps_attempted": 376,
        "control_steps_completed": 376,
    }
    measured = dict(
        measured_tcp=np.eye(4), measured_opening=0.25, limits=np.tile([-1.0, 1.0], (7, 1)), current_q=np.zeros(7)
    )
    return policy, selector, old, latest, receipt, measured, switches


@pytest.mark.parametrize("closing,source", [(True, "bridge"), (False, "latest")])
def test_actual_boundary_rebinds_bridge_or_latest_without_history_reset(monkeypatch, closing, source):
    policy, selector, old, latest, receipt, measured, switches = fixture(monkeypatch, closing=closing)
    decision, stop = transition.prepare_bridge(policy, selector, old, latest, receipt, **measured)
    assert stop is None and decision["source"] == source and selector.state()["staged"]
    assert receipt["bridge_selection"]["reference_audit"]["geometry_independently_verified"]
    assert switches[0][1]["offset"] == switches[0][1]["next_cursor"] == 16
    if closing:
        assert decision["openings"][0] == pytest.approx(0.2 * 7 / 8 + 0.9 / 8)


def test_prior_actual_buffer_loss_stops_before_reference_selection(monkeypatch):
    policy, selector, old, latest, receipt, measured, switches = fixture(monkeypatch)
    policy.last_inner_loss = True
    decision, stop = transition.prepare_bridge(policy, selector, old, latest, receipt, **measured)
    assert decision is None and stop["reason"] == "observed_joint_buffer_loss"
    assert selector.state()["mode"] == "held" and not selector.state()["staged"] and not switches


@pytest.mark.parametrize(
    "completed,returned,uncertain",
    [(0, True, False), (3, True, False), (8, True, False), (8, False, False), (0, False, True)],
)
def test_bridge_transaction_follows_completed_physical_execution(monkeypatch, completed, returned, uncertain):
    policy, selector, old, latest, receipt, measured, _ = fixture(monkeypatch)
    decision, stop = transition.prepare_bridge(policy, selector, old, latest, receipt, **measured)
    assert decision is not None and stop is None
    receipt["continuation_partial"] = {"periods": [{}] * completed}
    receipt["control_steps_completed"] += completed
    receipt["control_steps_attempted"] += completed + int(uncertain)
    transition.finish_bridge(selector, receipt, segment_returned=returned)
    state = selector.state()
    assert receipt["bridge_selection"]["completed_bridge_steps"] == completed
    assert state["staged"] is (uncertain or (completed == 8 and not returned))
    assert state["faulted"] is (0 < completed < 8 and not uncertain)
    if completed == 8 and returned and not uncertain:
        assert state["mode"] == "lockout"


def test_completed_history_comparison_detects_changes():
    expected = {"vectors": [1.0, 2.0], "nested": {"closing": True, "steps": 4}}
    transition.compare_history({"vectors": np.array([1.0, 2.0]), "nested": {"closing": True, "steps": 4}}, expected)
    with pytest.raises((ValueError, AssertionError)):
        transition.compare_history({"vectors": [1.0, 2.1], "nested": {"closing": True, "steps": 4}}, expected)
