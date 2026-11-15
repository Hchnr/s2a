"""The retrospective scorer cannot use a current or future measurement."""

import copy

import numpy as np
import pytest

from scripts.tracking.response_models import evaluate_action_conditioned_tcp as evaluation


def _period(step, predicted, actual, command=0.0):
    return {
        "step": step,
        "command": [command] * 6 + [0.0],
        "decision": {"quality": {"first_tcp_error": [predicted, 0, 0, 0, 0, 0], "verification": {"passed": True}}},
        "tcp_error": [actual, 0, 0, 0, 0, 0],
        "pose_caps_satisfied": abs(actual) <= 0.01,
    }


def test_current_observation_does_not_affect_forecast_but_updates_later_history():
    periods = [_period(0, 0, 0.002), _period(1, 0, 0.003, 0.4), _period(2, 0, 0, -0.4)]
    changed = copy.deepcopy(periods)
    changed[1]["tcp_error"][0] = 0.009
    before, after = evaluation.evaluate(periods), evaluation.evaluate(changed)
    for method in evaluation.METHODS:
        np.testing.assert_array_equal(before[1][method]["predicted_error"], after[1][method]["predicted_error"])
    assert before[2]["action_conditioned"]["predicted_error"][0] != after[2]["action_conditioned"]["predicted_error"][0]
    assert before[0]["action_conditioned"] is None
    assert [row["fit_history"] for row in before] == [0, 1, 2]


def test_summary_counts_false_safe_and_stop_on_shared_history():
    rows = evaluation.evaluate([_period(0, 0, 0), _period(1, 0, 0.012), _period(2, 0.012, 0)])
    result = evaluation.summarize(rows)
    assert result["recorded_periods"] == 3 and result["scored_periods"] == result["ridge_fits"] == 2
    assert result["methods"]["raw"]["false_safe"] == 1
    assert result["methods"]["raw"]["false_stop"] == 1
    assert {item["periods"] for item in result["methods"].values()} == {2}
    assert evaluation.evaluate([_period(0, 0, 0)])[0]["history_available"] is False


def test_scorer_rejects_missing_verification_or_changed_physical_gate():
    period = _period(0, 0, 0)
    period["decision"]["quality"]["verification"]["passed"] = False
    with pytest.raises(ValueError, match="independently checked"):
        evaluation.evaluate([period])
    period = _period(0, 0, 0)
    period["pose_caps_satisfied"] = False
    with pytest.raises(ValueError, match="physical pose gate"):
        evaluation.evaluate([period])
