"""Scalar event reconstruction detects timing and candidate mismatches."""

import copy

import numpy as np
import pytest

from s2a.research.wam.training.broader_recovery_events import event_candidates
from scripts.verification.wam.training import verify_broader_recovery_events as run


@pytest.mark.parametrize("periods", [96, 120, 180, 240])
def test_scalar_event_search_matches_vectorized_contract(periods):
    rng = np.random.default_rng(42 + periods)
    for _ in range(8):
        opening = rng.uniform(0.1, 1.0, size=periods + 1)
        success = rng.random(periods + 1) > 0.96
        expected = event_candidates(opening, success)
        actual = run.scalar_events(opening, success)
        assert actual == expected
        run.compare_events(actual, expected)


def test_scalar_event_audit_rejects_changed_counts_or_candidate_identity():
    opening = np.full(141, 0.8)
    success = np.zeros(141, dtype=bool)
    opening[101] = 0.2
    success[121:] = True
    expected = run.scalar_events(opening, success)
    changed = copy.deepcopy(expected)
    changed["low_opening"]["covered"] += 1
    with pytest.raises(ValueError, match="event counts"):
        run.compare_events(expected, changed)
    changed = copy.deepcopy(expected)
    changed["candidates"][0]["anchor_step"] += 1
    with pytest.raises(ValueError, match="selected event anchor"):
        run.compare_events(expected, changed)
