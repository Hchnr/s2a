"""Replay boundaries and the timing of completed quality feedback."""

import sys

import numpy as np
import pytest

from s2a.research.joint_control.preview.quality_feedback_refiner import QualitySelectionStopped
from s2a.research.response_models.causal.causal_inner_error import CausalInnerError
from scripts.simulation.runtime import run_early_quality_feedback as run


def test_checker_replays_zero_actions_and_copies_only_recorded_decision_stop(monkeypatch):
    for module in (run.QUALITY_MODULE, run.baseline.POLICY_MODULE, run.baseline.PREVIEW_MODULE):
        monkeypatch.delitem(sys.modules, module, raising=False)
    action = run.checker_action([])
    assert callable(action)
    expected = {"reason": "quality_no_witness", "at_step": 43, "next_action_dispatched": False}
    checked = run.reconcile_stop(
        {"periods": [], "stop": None}, expected, phase="checker", mode="quality", start_step=43
    )
    assert checked["stop"] == expected
    assert checked["decision_stop_recorded_not_recomputed"] is True


def test_checker_rejects_a_stop_at_the_wrong_step_or_a_changed_physical_stop():
    expected = {"reason": "quality_no_witness", "at_step": 44, "next_action_dispatched": False}
    with pytest.raises(ValueError, match="Decision stop position"):
        run.reconcile_stop({"periods": [], "stop": None}, expected, phase="checker", mode="quality", start_step=43)
    with pytest.raises(ValueError, match="physical stop"):
        run.reconcile_stop(
            {"periods": [], "stop": None},
            {"reason": "actual_inner_joint_limit_violation"},
            phase="checker",
            mode="quality",
            start_step=43,
        )


def test_feedback_updates_error_only_after_policy_observes_actual_period():
    history = CausalInnerError()
    history.stage(43, np.zeros((25, 7)))
    receipt = {
        "continuation_partial": {
            "periods": [{"step": 43, "actual_inner": [{"q": np.full(7, 0.002)} for _ in range(25)]}]
        }
    }

    class Policy:
        def select(self, *_args):
            raise QualitySelectionStopped("quality_no_witness", "no_witness")

        def observe(self, _env, _command, offset):
            assert offset == 0 and history.snapshot()["completed_periods"] == 0
            return {"observed": True}

    action, completed = run.quality_callbacks(Policy(), history, receipt, 43)
    assert action(None, None, None, None, 0)["stop"]["reason"] == "quality_no_witness"
    assert history.snapshot()["completed_periods"] == 0
    result = completed(None, np.zeros(7), 0)
    assert result["observed"] is True and result["past_error_after_completion"]["completed_periods"] == 1
    np.testing.assert_allclose(result["past_error_after_completion"]["inner_position_error_rad"], 0.002)


def test_checker_requires_saved_actions_and_a_fresh_process(monkeypatch):
    for module in (run.QUALITY_MODULE, run.baseline.POLICY_MODULE, run.baseline.PREVIEW_MODULE):
        monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ValueError, match="Saved actions"):
        run.checker_action(None)
    monkeypatch.setitem(sys.modules, run.QUALITY_MODULE, object())
    with pytest.raises(ValueError, match="must not import"):
        run.checker_action([])


def test_registered_cases_restore_only_the_bounded_completed_prefix():
    inputs = run.load_inputs()
    cases = [run.baseline.completed_case(index, inputs) for index in (2, 6)]
    assert [case["spec"]["prefix_steps"] for case in cases] == [43, 52]
    assert [case["completed_history"].completed_periods for case in cases] == [43, 52]
