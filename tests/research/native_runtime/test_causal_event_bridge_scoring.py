"""Actual endpoint scores must distinguish the bridge from both input targets."""

import copy

import numpy as np
import pytest

from s2a.geometry import pose_error
from scripts.simulation.runtime import evaluate_causal_event_bridge as scoring


def fixture(monkeypatch):
    monkeypatch.setattr(scoring, "current_joint_site_state", lambda _env: (None, None, np.zeros((6, 7))))
    monkeypatch.setattr(scoring, "current_robot_dynamics", lambda _env, _jac: (None, None, np.eye(4)))
    grip = {"opening": 0.4, "qpos": np.array([0.1, -0.1]), "qvel": np.zeros(2), "motor_target": np.zeros(2)}
    monkeypatch.setattr(scoring.base, "measured_gripper", lambda _env: grip)
    old = np.tile(np.eye(4), (40, 1, 1))
    latest = np.tile(np.eye(4), (40, 1, 1))
    bridge = np.tile(np.eye(4), (40, 1, 1))
    old[:, 0, 3] = np.arange(40) / 100
    latest[:, 1, 3] = np.arange(40) / 100
    bridge[:, 2, 3] = np.arange(40) / 100
    case = {
        "spec": {"prefix_steps": 360},
        "controller": {"source_from_site": np.eye(4)},
        "tool": np.eye(4),
        "references": {
            "remembered": {"poses": old, "openings": np.full(40, 0.1), "cursor": 16},
            "latest": {"poses": latest, "openings": np.full(40, 0.9), "cursor": 0},
        },
    }
    return case, {"poses": bridge, "openings": np.full(40, 0.5)}


def test_scores_three_same_endpoint_targets_and_actual_opening(monkeypatch):
    case, reference = fixture(monkeypatch)
    own_error = pose_error(reference["poses"][8], np.eye(4))
    result = scoring.score_completed(None, case, reference, 8, own_error)
    assert result["selected_absolute_endpoint"] == 369
    np.testing.assert_allclose(result["reference_errors"]["remembered"][:3], [0.24, 0, 0])
    np.testing.assert_allclose(result["reference_errors"]["latest"][:3], [0, 0.08, 0])
    np.testing.assert_allclose(result["reference_errors"]["bridge"][:3], [0, 0, 0.08])
    assert result["opening_errors"] == pytest.approx({"remembered": 0.3, "latest": -0.5, "bridge": -0.1})
    scoring.score_completed(None, case, reference, 8, own_error, expected=copy.deepcopy(result))


def test_rejects_a_mismatched_owned_or_replayed_bridge_error(monkeypatch):
    case, reference = fixture(monkeypatch)
    own_error = pose_error(reference["poses"][8], np.eye(4))
    with pytest.raises(AssertionError):
        scoring.score_completed(None, case, reference, 8, own_error + 0.01)
    expected = scoring.score_completed(None, case, reference, 8, own_error)
    expected["reference_errors"]["latest"][0] += 0.001
    with pytest.raises(AssertionError):
        scoring.score_completed(None, case, reference, 8, own_error, expected=expected)


def test_rejects_noncompleted_offset(monkeypatch):
    case, reference = fixture(monkeypatch)
    with pytest.raises(ValueError):
        scoring.score_completed(None, case, reference, 16, np.zeros(6))
