"""Action-conditioned corrections use only completed, per-trajectory history."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.response_models.causal.action_conditioned_tcp_residual import CausalActionTCPResidual


def test_candidate_queries_are_read_only_and_only_matching_observation_updates():
    model = CausalActionTCPResidual()
    prediction, action = np.zeros(6), np.zeros(6)
    first = model.stage(10, prediction, action)
    assert first["history_available"] is False and first["action_conditioned"] is None
    with pytest.raises(ValueError, match="previously staged action"):
        model.predict(10, prediction, action)
    with pytest.raises(ValueError, match="matching prediction"):
        model.complete(11, np.zeros(6))
    model.complete(10, [0.002, -0.001, 0.003, 0.02, -0.01, 0.03])
    first_candidate = model.predict(11, prediction, np.full(6, 0.1))
    second_candidate = model.predict(11, prediction, np.full(6, -0.1))
    assert first_candidate["completed_history"] == second_candidate["completed_history"] == 1
    np.testing.assert_allclose(first_candidate["action_conditioned"][:3], [0.002, -0.001, 0.003], atol=1e-12)
    first_candidate["action_conditioned"][:] = 99
    np.testing.assert_allclose(model.predict(11, prediction, action)["action_conditioned"][:3], [0.002, -0.001, 0.003])
    model.stage(11, prediction, action)
    with pytest.raises(ValueError, match="previously staged action"):
        model.stage(12, prediction, action)
    model.complete(11, np.zeros(6))
    with pytest.raises(ValueError, match="consecutive"):
        model.predict(13, prediction, action)


def test_fixed_window_ridge_matches_independent_augmented_least_squares():
    model, examples = CausalActionTCPResidual(), []
    for step in range(30):
        action = np.array([np.sin(step / (axis + 2)) for axis in range(6)]) * 0.7
        residual = np.array([action[0] - 0.2 * action[1], action[2] + 0.3, -action[3] + 0.1 * action[5]]) * 0.001
        predicted = np.array([0.001, -0.002, 0.003, 0.0, 0.0, 0.0])
        model.stage(step, predicted, action)
        model.complete(step, predicted + np.r_[residual, [0.0, 0.0, 0.0]])
        examples.append((action, residual))
    query = np.array([0.2, -0.4, 0.1, 0.3, 0.4, -0.1])
    forecast = model.predict(30, np.zeros(6), query)
    x = np.stack([item[0] for item in examples[-20:]])
    y = np.stack([item[1] for item in examples[-20:]])
    mx, my = x.mean(axis=0), y.mean(axis=0)
    augmented_x = np.vstack(((x - mx) / np.sqrt(20), np.eye(6)))
    augmented_y = np.vstack(((y - my) / np.sqrt(20), np.zeros((6, 3))))
    coefficients = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    np.testing.assert_allclose(forecast["action_conditioned"][:3], my + (query - mx) @ coefficients, atol=1e-12)
    assert forecast["fit_history"] == 20 and forecast["completed_history"] == 30
    assert forecast["ridge_fit_performed"] is True and forecast["future_guarantee"] is False


def test_rotation_uses_existing_noncommuting_so3_innovation():
    model = CausalActionTCPResidual()
    desired0 = Rotation.from_euler("xyz", [0.3, -0.2, 0.4])
    predicted0 = Rotation.from_euler("xyz", [-0.1, 0.4, 0.2])
    discrepancy = Rotation.from_euler("xyz", [0.08, -0.04, 0.12])
    actual0 = discrepancy * predicted0
    p0 = np.r_[np.zeros(3), (desired0 * predicted0.inv()).as_rotvec()]
    a0 = np.r_[np.zeros(3), (desired0 * actual0.inv()).as_rotvec()]
    model.stage(0, p0, np.zeros(6))
    model.complete(0, a0)
    desired1 = Rotation.from_euler("xyz", [-0.2, 0.25, -0.1])
    predicted1 = Rotation.from_euler("xyz", [0.4, 0.1, -0.3])
    p1 = np.r_[np.zeros(3), (desired1 * predicted1.inv()).as_rotvec()]
    corrected = model.predict(1, p1, np.zeros(6))["action_conditioned"]
    expected = desired1 * (discrepancy * predicted1).inv()
    np.testing.assert_allclose(Rotation.from_rotvec(corrected[3:]).as_matrix(), expected.as_matrix(), atol=1e-12)


def test_invalid_action_and_ambiguous_rotation_are_rejected_before_staging():
    model = CausalActionTCPResidual()
    with pytest.raises(ValueError, match="controller box"):
        model.stage(0, np.zeros(6), [1.01, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="pi ambiguity"):
        model.stage(0, [0, 0, 0, np.pi, 0, 0], np.zeros(6))
    assert model.stage(0, np.zeros(6), np.zeros(6))["completed_history"] == 0
