"""Correct-reference windows must share time and never expose recorded actions."""

import numpy as np
import pytest

from s2a.references.oracle_timeline import OracleTimeline


def windows(periods):
    size = 8 * ((periods + 7) // 8) + 32
    poses = np.repeat(np.eye(4)[None, :, :], size, axis=0)
    poses[:, 0, 3] = np.arange(size) * 0.001
    opening = np.linspace(0.1, 0.9, size)
    return [
        dict(
            step=index,
            effective_targets_tcp=poses[index : index + 40],
            openings=opening[index : index + 40],
            command="excluded",
        )
        for index in range(0, periods, 8)
    ]


def test_timeline_joins_overlapping_targets_and_marks_terminal_hold():
    source = windows(19)
    timeline = OracleTimeline(source, periods=19)
    poses, openings, metadata = timeline.window(15)
    assert metadata == dict(start_step=15, reference_endpoints=4, terminal_hold_endpoints=36)
    np.testing.assert_allclose(poses[:4, 0, 3], np.arange(15, 19) * 0.001)
    np.testing.assert_array_equal(poses[4:], np.repeat(poses[3:4], 36, axis=0))
    np.testing.assert_array_equal(openings[4:], np.repeat(openings[3:4], 36))
    poses[0, 0, 3] = 99
    assert timeline.window(15)[0][0, 0, 3] != 99
    assert not hasattr(timeline, "command")


@pytest.mark.parametrize("field", ["step", "pose", "opening"])
def test_timeline_rejects_misaligned_or_nonmatching_windows(field):
    source = windows(19)
    if field == "step":
        source[1]["step"] = 9
    elif field == "pose":
        source[1]["effective_targets_tcp"] = source[1]["effective_targets_tcp"].copy()
        source[1]["effective_targets_tcp"][0, 0, 3] += 1e-4
    else:
        source[1]["openings"] = source[1]["openings"].copy()
        source[1]["openings"][0] += 0.01
    with pytest.raises(ValueError):
        OracleTimeline(source, periods=19)


def test_timeline_rejects_cursor_outside_recorded_periods():
    timeline = OracleTimeline(windows(19), periods=19)
    for cursor in (-1, 19, 1.0):
        with pytest.raises(ValueError):
            timeline.window(cursor)


def test_timeline_accepts_short_final_windows_and_rejects_missing_endpoints():
    source = windows(51)
    for row in source:
        count = min(40, 51 - row["step"])
        row["effective_targets_tcp"] = row["effective_targets_tcp"][:count]
        row["openings"] = row["openings"][:count]
    timeline = OracleTimeline(source, periods=51)
    poses, _, metadata = timeline.window(48)
    np.testing.assert_allclose(poses[:3, 0, 3], np.arange(48, 51) * 0.001)
    np.testing.assert_array_equal(poses[3:], np.repeat(poses[2:3], 37, axis=0))
    assert metadata["terminal_hold_endpoints"] == 37
    source[-1]["effective_targets_tcp"] = source[-1]["effective_targets_tcp"][:-1]
    source[-1]["openings"] = source[-1]["openings"][:-1]
    with pytest.raises(ValueError, match="cover its remaining"):
        OracleTimeline(source, periods=51)
