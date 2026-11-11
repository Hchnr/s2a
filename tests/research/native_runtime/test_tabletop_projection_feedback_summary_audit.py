"""Replay target offsets must be reconstructed, never silently defaulted."""

import numpy as np
import pytest

from scripts.verification.reference_ports.audit_tabletop_projection_feedback_summary import replay_reference_shift


def test_reference_shift_uses_the_original_and_replayed_targets():
    target, selected = np.eye(4), np.eye(4)
    selected[2, 3] = 0.03
    producer = {
        "decision": {"original_target_tcp": target, "selected_target_tcp": selected, "reference_translation_m": 0.03}
    }
    checker = {"decision": {"selected_target_tcp": selected.copy()}}
    assert replay_reference_shift(producer, checker) == pytest.approx(0.03)
    checker["decision"]["selected_target_tcp"][2, 3] = 0.0
    with pytest.raises(AssertionError):
        replay_reference_shift(producer, checker)
