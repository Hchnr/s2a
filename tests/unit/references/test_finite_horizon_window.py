"""A saved finite reference cannot silently invent or shift future endpoints."""

import numpy as np
import pytest

from s2a.references.finite_horizon_window import remaining_h40


def test_each_segment_contains_remaining_model_targets_then_terminal_hold():
    poses = np.tile(np.eye(4), (40, 1, 1))
    poses[:, 0, 3] = np.arange(40) / 100
    openings = np.linspace(1, 0, 40)
    for consumed in (0, 8, 16, 24, 32):
        window, opening, receipt = remaining_h40(poses, openings, consumed=consumed)
        count = 40 - consumed
        np.testing.assert_array_equal(window[:count], poses[consumed:])
        np.testing.assert_array_equal(opening[:count], openings[consumed:])
        np.testing.assert_array_equal(window[count:], np.repeat(poses[-1:], consumed, axis=0))
        np.testing.assert_array_equal(opening[count:], np.repeat(openings[-1:], consumed))
        assert receipt == dict(generated_endpoints=count, terminal_hold_endpoints=consumed)
        window[0, 0, 3] = -1
        assert poses[consumed, 0, 3] >= 0


@pytest.mark.parametrize("consumed", [-8, 1, 40, True])
def test_invalid_or_exhausted_segment_boundary_is_rejected(consumed):
    with pytest.raises(ValueError, match="aligned consumed prefix"):
        remaining_h40(np.tile(np.eye(4), (40, 1, 1)), np.ones(40), consumed=consumed)
