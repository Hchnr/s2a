"""The independent checker can only replay the saved commands."""

import sys

import numpy as np
import pytest

from scripts.simulation.runtime import run_early_refiner_feedback as feedback


def test_checker_replays_saved_action_without_importing_decision_modules(monkeypatch):
    monkeypatch.delitem(sys.modules, feedback.POLICY_MODULE, raising=False)
    monkeypatch.delitem(sys.modules, feedback.PREVIEW_MODULE, raising=False)
    recorded = [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
    action = feedback.checker_action([{"command": recorded}])
    result = action(None, None, None, None, 0)
    np.testing.assert_array_equal(result["command"], recorded)
    assert result["newly_selected"] is False
    assert feedback.POLICY_MODULE not in sys.modules
    assert feedback.PREVIEW_MODULE not in sys.modules


def test_checker_rejects_missing_saved_actions():
    with pytest.raises(ValueError, match="Saved actions"):
        feedback.checker_action(None)
