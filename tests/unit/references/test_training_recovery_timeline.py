"""The training oracle retains timing, tool geometry and native endpoints."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.references.training_recovery_timeline import training_recovery_timeline


@pytest.mark.parametrize("offset", [0.0, 0.02, -0.02])
def test_offset_return_and_native_targets_keep_the_fixed_tcp_contract(offset):
    source = np.repeat(np.eye(4)[None], 46, axis=0)
    source[:, :3, :3] = Rotation.from_euler("z", 0.4).as_matrix()
    source[:, 1, 3] = np.arange(len(source)) * 0.001
    opening = np.linspace(0.8, 0.2, len(source))
    tool = np.eye(4)
    tool[:3, 3] = [0.03, -0.02, 0.1]
    source_copy, opening_copy = source.copy(), opening.copy()
    timeline, meta = training_recovery_timeline(source, opening, tool, offset_m=offset)
    assert meta["periods"] == 16 + 40 + len(source) - 1
    assert meta["physical_labels_from_reference"] is False
    first, first_opening, _ = timeline.window(0)
    initial = source[0] @ tool
    np.testing.assert_allclose(first[0, :3, 3], initial[:3, 3] + [offset / 16, 0, 0], atol=1e-12)
    np.testing.assert_allclose(timeline.window(15)[0][0, :3, 3], initial[:3, 3] + [offset, 0, 0], atol=1e-12)
    np.testing.assert_allclose(timeline.window(55)[0][0], initial, atol=1e-12)
    np.testing.assert_allclose(timeline.window(56)[0][0], source[1] @ tool, atol=1e-12)
    assert first_opening[0] == opening[0] and timeline.window(56)[1][0] == opening[1]
    tail, tail_opening, tail_meta = timeline.window(timeline.periods - 1)
    np.testing.assert_allclose(tail, np.repeat((source[-1] @ tool)[None], 40, axis=0), atol=1e-12)
    assert np.all(tail_opening == opening[-1]) and tail_meta["terminal_hold_endpoints"] == 39
    np.testing.assert_array_equal(source, source_copy)
    np.testing.assert_array_equal(opening, opening_copy)


def test_recovery_rejects_invalid_tool_and_short_native_reference():
    source = np.repeat(np.eye(4)[None], 41, axis=0)
    with pytest.raises(ValueError):
        training_recovery_timeline(source, np.ones(41), np.ones((4, 4)), offset_m=0.0)
    with pytest.raises(ValueError, match="full source endpoints"):
        training_recovery_timeline(source[:40], np.ones(40), np.eye(4), offset_m=0.0)
