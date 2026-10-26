import numpy as np
import pytest
from scipy.linalg import expm

from s2a.frozen_osc_response import goal_response


def test_decoupled_response_matches_six_independent_pd_systems():
    kp = np.arange(6) + 100.0
    kd = 2 * np.sqrt(kp)
    result, _, b = goal_response(np.diag(np.arange(6) + 1.0), np.eye(6), kp, kd, 0.05)
    np.testing.assert_allclose(b, np.eye(6), atol=1e-14)
    expected = np.zeros((12, 6))
    for i in range(6):
        transition = expm(np.array([[0, 1], [-kp[i], -kd[i]]]) * 0.05)
        expected[i, i] = 1 - transition[0, 0]
        expected[6 + i, i] = -transition[1, 0]
    np.testing.assert_allclose(result, expected, atol=1e-13)


def test_cross_inertia_induces_rotation_from_translation_goal():
    j = np.eye(6)
    j[3, 0] = 0.4
    response, _, b = goal_response(np.eye(6), j, np.ones(6) * 100, np.ones(6) * 20, 0.05)
    assert b[3, 0] > 0 and response[3, 0] > 0 and response[9, 0] > 0
    uncoupled, _, _ = goal_response(np.eye(6), np.eye(6), np.ones(6) * 100, np.ones(6) * 20, 0.05)
    assert uncoupled[3, 0] == uncoupled[9, 0] == 0


def test_invalid_mass_is_rejected():
    with pytest.raises(np.linalg.LinAlgError):
        goal_response(-np.eye(6), np.eye(6), np.ones(6), np.ones(6), 0.05)
