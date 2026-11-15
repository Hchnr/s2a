"""Frozen comparisons must use the first arm change after matching prior actions."""

import numpy as np
import pytest

from scripts.verification.trajectory_search.audit_cached_seed_first_divergence import first_divergence


def test_first_divergence_rejects_a_gripper_change_at_the_same_step():
    original = np.zeros((5, 7))
    candidate = original.copy()
    candidate[2, 0] = 0.01
    assert first_divergence(original, candidate) == 2
    candidate[2, 6] = 1.0
    with pytest.raises(ValueError, match="matching gripper"):
        first_divergence(original, candidate)


def test_first_divergence_does_not_treat_roundoff_as_a_new_action():
    original = np.zeros((4, 7))
    candidate = original.copy()
    candidate[0, 0] = 1e-12
    candidate[3, 1] = 0.02
    assert first_divergence(original, candidate) == 3
