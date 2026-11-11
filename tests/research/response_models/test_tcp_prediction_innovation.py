"""Completed observations alone may change frame-consistent TCP predictions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.response_models.causal.tcp_prediction_innovation import CausalTCPPredictionInnovation


def test_noncommuting_world_rotation_innovation_is_composed_on_so3():
    history = CausalTCPPredictionInnovation()
    desired0 = Rotation.from_euler("xyz", [0.3, -0.2, 0.4])
    predicted0 = Rotation.from_euler("xyz", [-0.1, 0.4, 0.2])
    discrepancy = Rotation.from_euler("xyz", [0.08, -0.04, 0.12])
    actual0 = discrepancy * predicted0
    predicted_error0 = np.r_[[0.01, 0.02, -0.01], (desired0 * predicted0.inv()).as_rotvec()]
    actual_error0 = np.r_[[0.012, 0.015, -0.007], (desired0 * actual0.inv()).as_rotvec()]
    first = history.stage(80, predicted_error0)
    assert first["history_available"] is False and first["previous_innovation"] is None
    history.complete(80, actual_error0)

    desired1 = Rotation.from_euler("xyz", [-0.2, 0.25, -0.1])
    predicted1 = Rotation.from_euler("xyz", [0.4, 0.1, -0.3])
    predicted_error1 = np.r_[[0.03, -0.01, 0.02], (desired1 * predicted1.inv()).as_rotvec()]
    expected_rotation = desired1 * (discrepancy * predicted1).inv()
    second = history.stage(81, predicted_error1)
    np.testing.assert_allclose(
        Rotation.from_rotvec(second["previous_innovation"][3:]).as_matrix(), expected_rotation.as_matrix(), atol=1e-12
    )
    np.testing.assert_allclose(second["previous_innovation"][:3], [0.032, -0.015, 0.023], atol=1e-12)
    assert second["future_guarantee"] is False


def test_history_is_updated_only_after_matching_completed_period():
    history = CausalTCPPredictionInnovation()
    predicted, actual = np.zeros(6), np.array([0.003, 0, 0, 0, 0.04, 0])
    before = history.stage(10, predicted)
    with pytest.raises(ValueError, match="unobserved"):
        history.stage(11, predicted)
    with pytest.raises(ValueError, match="matching prediction"):
        history.complete(11, actual)
    assert before["completed_history"] == history.completed == 0
    history.complete(10, actual)
    after = history.stage(11, predicted)
    assert after["completed_history"] == 1
    assert after["past_maximum_bound_m"] == pytest.approx(0.003)
    assert after["past_maximum_bound_rad"] == pytest.approx(0.04)
    np.testing.assert_allclose(after["previous_innovation"], actual, atol=1e-12)


def test_nonconsecutive_or_pi_ambiguous_updates_are_rejected():
    history = CausalTCPPredictionInnovation()
    with pytest.raises(ValueError, match="pi ambiguity"):
        history.stage(1, [0, 0, 0, np.pi, 0, 0])
    history.stage(1, np.zeros(6))
    history.complete(1, np.zeros(6))
    with pytest.raises(ValueError, match="consecutive"):
        history.stage(3, np.zeros(6))
