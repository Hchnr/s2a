import numpy as np
import pytest
from scipy.spatial.distance import cdist

from s2a.training_support import TrainingSupport, blend_translation, calibrate_support


def test_calibration_excludes_whole_trajectory_and_matches_brute_force():
    points = np.array([[0, 0], [0.01, 0], [3, 0], [3.01, 0], [0, 4], [0.01, 4]])
    groups = np.repeat([0, 1, 2], 2)
    expected = cdist(points, points)
    expected[groups[:, None] == groups[None, :]] = np.inf
    expected = expected.min(axis=1)
    result = calibrate_support(points, groups)
    np.testing.assert_allclose(result["cross_trajectory_distances"], expected)
    assert result["radius"] == pytest.approx(np.quantile(expected, 0.95))
    assert result["radius"] > 3  # Near-duplicate within-trajectory rows cannot leak into calibration.


def test_support_weights_and_input_snapshot():
    points = np.array([[0.0, 0], [0, 1]])
    support = TrainingSupport(points, 2)
    points[:] = 100
    assert support.query([0, 0])["learned_weight"] == 1
    assert support.query([2, 0])["learned_weight"] == 1
    assert support.query([3, 0])["learned_weight"] == pytest.approx(0.5)
    assert support.query([4, 0])["learned_weight"] == 0
    with pytest.raises(ValueError):
        support.query([np.nan, 0])


def test_switch_to_physical_candidate_preserves_bounds_rate_and_previous_input():
    previous = np.array([0.9, -0.9, 0.0])
    original = previous.copy()
    bounds = dict(lower=-np.ones(3), upper=np.ones(3))
    out = blend_translation([-1, 1, 1], [2, -2, -2], previous, weight=0, **bounds)
    np.testing.assert_allclose(out, [1, -1, -0.25])
    np.testing.assert_array_equal(previous, original)
    middle = blend_translation([0.2, 0.1, 0], [-0.2, -0.1, 0.4], np.zeros(3), weight=0.5, **bounds)
    np.testing.assert_allclose(middle, [0, 0, 0.2])
    with pytest.raises(ValueError):
        blend_translation([0, 0, 0], [0, 0, 0], [2, 0, 0], weight=1, **bounds)


def test_degenerate_calibration_and_invalid_weights_rejected():
    with pytest.raises(ValueError, match="trajectories"):
        calibrate_support(np.zeros((2, 3)), [0, 0])
    with pytest.raises(ValueError, match="nondegenerate"):
        calibrate_support(np.zeros((2, 3)), [0, 1])
    with pytest.raises(ValueError, match="weight"):
        blend_translation([0, 0, 0], [0, 0, 0], [0, 0, 0], weight=np.nan, lower=-np.ones(3), upper=np.ones(3))
