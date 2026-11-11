"""Exercise exact generated-chunk boundaries and independent support checks."""

import numpy as np
import pytest

from s2a.research.gripper.causal_opening_event_metrics import select_threshold
from s2a.research.gripper.supported_opening_event import future_closure_support, supported_probability
from scripts.verification.wam.training.verify_supported_opening_event import verify_support, verify_supported_selection
from tests.research.gripper.test_causal_opening_event_metrics import config, data


def test_only_later_endpoints_inside_the_sixteen_step_suffix_support_closure():
    raw = np.full((3, 40), 0.8)
    raw[0, 16] = raw[1, 17] = raw[2, 0] = 0.1
    mask = future_closure_support(raw)
    assert mask[0, 0] and not mask[1, 0] and mask[1, 1]
    assert not mask[2].any()
    assert not mask[:, -1].any()
    raw[2, 1] = 0.35
    assert not future_closure_support(raw)[2, 0]
    assert verify_support(raw, future_closure_support(raw), steps=16, closed_below=0.35)["queries"] == 3


def test_support_masks_only_probabilities_without_mutating_inputs():
    raw = np.full((2, 40), 0.8)
    raw[0, 10] = 0.1
    probability = np.full((2, 40), 0.9)
    original_raw, original_probability = raw.copy(), probability.copy()
    result = supported_probability(raw, probability)
    assert (result[0, :10] == 0.9).all() and not result[0, 10:].any()
    assert not result[1].any()
    np.testing.assert_array_equal(raw, original_raw)
    np.testing.assert_array_equal(probability, original_probability)


@pytest.mark.parametrize("steps", [0, 40, True])
def test_rejects_invalid_support_horizon(steps):
    with pytest.raises(ValueError, match="future steps"):
        future_closure_support(np.ones((1, 40)), steps=steps)


@pytest.mark.parametrize(
    "raw,probability", [(np.ones(39), np.ones(39)), (np.ones(40), np.ones(39)), (np.ones(40), np.full(40, 1.1))]
)
def test_rejects_invalid_arrays(raw, probability):
    with pytest.raises(ValueError):
        supported_probability(raw, probability)


def test_independent_checker_recomputes_support_and_registered_gate():
    raw, truth, probability, tasks = data()
    raw[0, 8] = raw[3, 8] = 0.1
    mask = future_closure_support(raw)
    masked = supported_probability(raw, probability)
    selected = select_threshold(raw, truth, masked, tasks, config())
    assert (
        verify_supported_selection(raw, truth, probability, tasks, mask, config(), selected)["selection"][
            "selected_threshold"
        ]
        == 0.9
    )
    mask[0, 0] = False
    with pytest.raises(ValueError, match="support differs"):
        verify_supported_selection(raw, truth, probability, tasks, mask, config(), selected)
