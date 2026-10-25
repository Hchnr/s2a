"""Causal endpoint alignment, pose disagreement, and episode-age safeguards."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.reference_commitment import CommitmentConfig, select_committed_prefix

CONFIG = CommitmentConfig(0.02, 0.1, 0.03, 0.15, 0.9)


def data():
    old = np.tile(np.eye(4), (40, 1, 1))
    old[:, 0, 3] = np.arange(1, 41) * 0.001
    latest = old.copy()
    latest[:32] = old[8:]
    old_opening = np.zeros(40)
    old_opening[15:] = 1
    return old, old_opening, latest, np.zeros(40)


def choose(values, **kwargs):
    return select_committed_prefix(
        *values,
        previous_step=0,
        current_step=kwargs.pop("current_step", 8),
        measured_tcp=kwargs.pop("measured_tcp", values[0][7]),
        prefix_count=8,
        config=CONFIG,
        **kwargs,
    )


def test_retains_same_absolute_endpoints_and_whole_pose_opening_pair():
    values = data()
    poses, openings, provenance = choose(values)
    np.testing.assert_array_equal(poses, values[0][8:16])
    np.testing.assert_array_equal(openings, values[1][8:16])
    assert provenance["source"] == "previous"
    assert provenance["source_indices"] == list(range(8, 16))
    poses[:] = 0
    assert values[0][8, 3, 3] == 1  # Returned arrays cannot mutate stored generation.


@pytest.mark.parametrize("failure", ["position", "rotation", "measured_position", "measured_rotation"])
def test_rejects_old_phase_when_pose_or_current_tracking_disagrees(failure):
    values = data()
    measured = values[0][7].copy()
    if failure == "position":
        values[2][3, 1, 3] = 0.1  # One inconsistent endpoint must reject the whole prefix.
    elif failure == "rotation":
        values[2][3, :3, :3] = Rotation.from_euler("z", 179, degrees=True).as_matrix()
    elif failure == "measured_position":
        measured[1, 3] = 0.1
    else:
        measured[:3, :3] = Rotation.from_euler("x", 0.5).as_matrix()
    poses, openings, provenance = choose(values, measured_tcp=measured)
    assert provenance["source"] == "latest"
    np.testing.assert_array_equal(poses, values[2][:8])
    np.testing.assert_array_equal(openings, values[3][:8])


@pytest.mark.parametrize("current_step", [0, 4, 16, 100])
def test_rejects_duplicate_misaligned_or_stale_previous_plan(current_step):
    assert choose(data(), current_step=current_step)[2]["source"] == "latest"


def test_latest_opening_phase_is_not_replaced():
    values = data()
    values[3][5] = 1
    assert choose(values)[2]["source"] == "latest"


def test_future_unused_endpoints_cannot_change_current_selection():
    values = data()
    expected = choose(values)
    values[0][16:, 1, 3] = 100
    values[1][16:] = 0.5
    values[2][8:, 1, 3] = -100
    values[3][8:] = 1
    actual = choose(values)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
