"""TCP error estimates use only completed past and the current model output."""

import copy

import numpy as np
import pytest

from s2a.infrastructure.publication import result_json
from scripts.tracking.response_models import audit_causal_tcp_prediction as audit


def periods():
    output = []
    for index in range(4):
        actual = [0.002 * index, 0, 0, 0, 0.01 * index, 0]
        output.append(
            {
                "step": 80 + index,
                "decision": {"quality": {"first_tcp_error": [0, 0, 0, 0, 0, 0], "verification": {"passed": True}}},
                "tcp_error": actual,
                "pose_caps_satisfied": True,
                "task_success": False,
            }
        )
    return output


def test_future_observations_and_task_labels_cannot_change_past_estimates():
    initial = periods()
    changed = copy.deepcopy(initial)
    changed[3]["tcp_error"][0] = 0.03
    changed[3]["pose_caps_satisfied"] = False
    for row in changed:
        row["task_success"] = True
    original, altered = audit.evaluate(initial), audit.evaluate(changed)
    assert result_json(original[:3]) == result_json(altered[:3])
    np.testing.assert_array_equal(
        original[3]["previous_innovation"]["predicted_error"], altered[3]["previous_innovation"]["predicted_error"]
    )


def test_no_history_is_excluded_and_gate_counts_are_explicit():
    rows = audit.evaluate(periods())
    summary = audit.summarize(rows)
    assert rows[0]["history_available"] is False
    assert rows[0]["previous_innovation"] is None and rows[0]["past_maximum_bound"] is None
    assert all(item["periods"] == 3 for item in summary["candidates"].values())
    assert summary["candidates"]["raw"]["false_safe"] == 0
    assert audit._safe(np.array([0.01 + 5e-11, 0, 0, 0, 0.05, 0]))
    assert not audit._safe(np.array([0.01 + 2e-10, 0, 0, 0, 0.05, 0]))


def test_unverified_or_mislabeled_physical_period_is_rejected():
    rows = periods()
    rows[0]["decision"]["quality"]["verification"]["passed"] = False
    with pytest.raises(ValueError, match="checked model prediction"):
        audit.evaluate(rows)
    rows = periods()
    rows[0]["pose_caps_satisfied"] = False
    with pytest.raises(ValueError, match="pose gate"):
        audit.evaluate(rows)
