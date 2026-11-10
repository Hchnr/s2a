"""Both references are scored against the same actual endpoint and time."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.joint_control.tracking.replan_reference_scoring import score_reference_pair


def test_aligned_reference_pair_uses_the_completed_endpoint():
    old = np.tile(np.eye(4), (8, 1, 1))
    new = old.copy()
    old[3, 0, 3] = 0.01
    new[3, :3, :3] = Rotation.from_rotvec([0, 0, 0.1]).as_matrix()
    actual = np.eye(4)
    actual[1, 3] = 0.002
    score = score_reference_pair(actual, old, new, 3)
    np.testing.assert_allclose(score["old"][:3], [0.01, -0.002, 0], atol=1e-12)
    np.testing.assert_allclose(score["new"][:3], [0, -0.002, 0], atol=1e-12)
    np.testing.assert_allclose(score["new"][3:], [0, 0, 0.1], atol=1e-12)


def test_future_reference_changes_cannot_change_current_score():
    old = np.tile(np.eye(4), (8, 1, 1))
    new = old.copy()
    before = score_reference_pair(np.eye(4), old, new, 2)
    old[3:, 0, 3], new[3:, 1, 3] = 3.0, -2.0
    after = score_reference_pair(np.eye(4), old, new, 2)
    for name in ("old", "new"):
        np.testing.assert_array_equal(before[name], after[name])


@pytest.mark.parametrize("offset", [-1, 8, 1.0, True])
def test_wrong_endpoint_is_rejected(offset):
    targets = np.tile(np.eye(4), (8, 1, 1))
    with pytest.raises(ValueError, match="completed offset"):
        score_reference_pair(np.eye(4), targets, targets, offset)


def test_missing_or_nonfinite_reference_is_rejected():
    targets = np.tile(np.eye(4), (8, 1, 1))
    with pytest.raises(ValueError):
        score_reference_pair(np.eye(4), targets[:7], targets, 0)
    targets[3, 0, 3] = float("nan")
    with pytest.raises(ValueError):
        score_reference_pair(np.eye(4), targets, targets, 0)
