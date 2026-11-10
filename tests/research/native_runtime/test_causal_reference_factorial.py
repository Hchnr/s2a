"""Component interventions must use available, aligned references only."""

import numpy as np
import pytest

from scripts.simulation.runtime.construct_causal_reference_factorial import construct_reference
from tests.research.wam.tracking.preview.test_causal_reference_bridge import bridge, references


def fixture(mode):
    old, old_open, _latest, _latest_open = references()
    blended = bridge(references())
    kwargs = {
        "old_step": 344,
        "origin_step": 360,
        "common_steps": 8,
        "continuation_steps": 16,
        "preview_steps": 8,
        "mode": mode,
    }
    return old, old_open, blended["poses"], blended["openings"], kwargs


@pytest.mark.parametrize("mode", ["bridge_pose_old_opening", "old_pose_bridge_opening"])
def test_same_time_components_and_old_horizon_boundary(mode):
    old, old_open, poses, openings, kwargs = fixture(mode)
    before = [x.copy() for x in (old, old_open, poses, openings)]
    result = construct_reference(old, old_open, poses, openings, **kwargs)
    assert result["old_aligned_horizon"] == 24
    np.testing.assert_array_equal(result["poses"][:8], old[16:24])
    np.testing.assert_array_equal(result["openings"][:8], old_open[16:24])
    np.testing.assert_array_equal(result["poses"][24:], poses[24:])
    np.testing.assert_array_equal(result["openings"][24:], openings[24:])
    if mode == "bridge_pose_old_opening":
        np.testing.assert_array_equal(result["poses"], poses)
        np.testing.assert_array_equal(result["openings"][:24], old_open[16:])
    else:
        np.testing.assert_array_equal(result["poses"][:24], old[16:])
        np.testing.assert_array_equal(result["openings"], openings)
    for actual, expected in zip((old, old_open, poses, openings), before, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "override",
    [
        {"old_step": 360},
        {"old_step": 300},
        {"origin_step": True},
        {"common_steps": 0},
        {"continuation_steps": 8},
        {"preview_steps": 0},
        {"preview_steps": 10},
        {"mode": "unregistered"},
    ],
)
def test_rejects_noncausal_or_unavailable_preview(override):
    old, old_open, poses, openings, kwargs = fixture("bridge_pose_old_opening")
    kwargs.update(override)
    with pytest.raises(ValueError):
        construct_reference(old, old_open, poses, openings, **kwargs)


def test_rejects_a_mismatched_common_prefix():
    old, old_open, poses, openings, kwargs = fixture("old_pose_bridge_opening")
    openings[7] += 0.001
    with pytest.raises(ValueError, match="common"):
        construct_reference(old, old_open, poses, openings, **kwargs)
