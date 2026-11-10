"""The stopped-state audit binds the independently replayed feedback results."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_quality_stop_feasibility as audit


def test_registered_stop_inputs_bind_two_completed_feedback_branches():
    inputs = audit.load_inputs()
    cases = inputs["cases"]
    assert [case["spec"]["case_index"] for case in cases] == [2, 6]
    assert [case["stop_inputs"]["stop_step"] for case in cases] == [48, 57]
    for case in cases:
        stop = case["stop_inputs"]
        assert stop["completed_periods"] == 5
        np.testing.assert_array_equal(stop["previous"], case["feedback_periods"][-1]["command"])
        np.testing.assert_array_equal(stop["state"], case["feedback_periods"][-1]["actual_state"])


def test_changed_feedback_result_is_rejected_before_model_construction(monkeypatch):
    changed = dict(audit.PRODUCER, sha256="0" * 64)
    monkeypatch.setattr(audit, "PRODUCER", changed)
    with pytest.raises(ValueError, match="Changed input"):
        audit.load_inputs()
