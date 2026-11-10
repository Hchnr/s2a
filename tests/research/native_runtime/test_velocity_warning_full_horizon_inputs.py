"""Earlier warning inputs bind the correct source and already-generated chunk."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_velocity_warning_full_horizon as audit


def test_earlier_inputs_use_first_causal_warning_and_current_chunk():
    inputs = audit.load_inputs()
    cases = inputs["cases"]
    assert [case["first_warning"]["endpoint"] for case in cases] == [39, 47]
    assert [case["intervention_spec"] for case in cases] == [
        {"cursor": 7, "chunk_index": 4},
        {"cursor": 7, "chunk_index": 5},
    ]
    for case in cases:
        decision = case["decision"]
        assert set(decision) == {
            "plan",
            "nominal_response",
            "mass",
            "jacobian",
            "q",
            "velocity",
            "site",
            "twist",
            "chunk_index",
            "cursor",
        }
        assert decision["plan"].shape == (8, 6)
        np.testing.assert_array_equal(case["feedback_periods"][0]["decision"]["refined_plan"], decision["plan"])
        assert case["first_warning"]["minimum_margin_rad"] >= 0.05


def test_changed_original_warning_audit_rejected(monkeypatch):
    monkeypatch.setattr(audit.early, "AUDIT", dict(audit.early.AUDIT, sha256="0" * 64))
    with pytest.raises(ValueError, match="Changed input"):
        audit.load_inputs()
