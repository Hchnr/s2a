"""Candidate-window unions must count unique real transitions only once."""

import copy

import numpy as np
import pytest

from scripts.verification.wam.training import verify_broader_recovery_event_union as run
from tests.research.wam.training.test_recovery_h40_contract import window_data


def test_union_counts_events_once_across_overlapping_h40_windows():
    _, normalization = window_data()
    normalization = copy.deepcopy(normalization)
    normalization["transform"]["scale"][9] = 1.0
    opening = np.full(181, 0.8)
    opening[101:121] = 0.2
    success = np.zeros(181, dtype=bool)
    success[151:] = True
    actions = np.zeros((181, 128))
    actions[:-1, 9] = opening[1:]
    data = dict(opening=opening, native_success=success, action128=actions)
    result = run.summarize_union(data, [89, 118], normalization)
    assert result["existing"]["unique_case_transition_rows"] == 96
    assert result["candidates"]["unique_case_transition_rows"] == 69
    assert result["union"]["unique_case_transition_rows"] == 158
    assert result["added"]["unique_case_transition_rows"] == 62
    assert result["existing"]["low_opening_endpoints"] == 0
    assert result["union"]["low_opening_endpoints"] == 20
    assert result["union"]["native_success_endpoints"] == 8
    assert result["union"]["clipped_total"] == 0
    with pytest.raises(ValueError, match="Unique full-horizon"):
        run.summarize_union(data, [89, 89], normalization)
    with pytest.raises(ValueError, match="Unique full-horizon"):
        run.summarize_union(data, [141], normalization)
