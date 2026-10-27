import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.trajectory_objective import opening_loss, position_loss


def test_endpoint_gradient_with_rotating_frame_and_nonuniform_weights():
    rng = np.random.default_rng(42)
    delta = rng.normal(size=(5, 3)) * 0.01
    rotations = Rotation.random(5, random_state=rng).as_matrix()
    initial = np.array([0.4, -0.1, 0.2])
    desired = initial + rng.normal(size=(5, 3)) * 0.02
    weights = np.array([3, 2, 2, 1, 1])
    loss, gradient = position_loss(delta, rotations, initial, desired, weights)
    numerical = np.zeros_like(delta)
    for index in np.ndindex(delta.shape):
        plus, minus = delta.copy(), delta.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (
            position_loss(plus, rotations, initial, desired, weights)[0]
            - position_loss(minus, rotations, initial, desired, weights)[0]
        ) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=1e-11, rtol=1e-8)
    assert loss > 0


def test_coherent_error_costs_more_than_cancellation():
    coherent = np.array([[0.1, 0, 0], [0.1, 0, 0]])
    cancel = coherent.copy()
    cancel[1] *= -1
    args = (np.tile(np.eye(3), (2, 1, 1)), np.zeros(3), np.zeros((2, 3)), np.ones(2))
    assert position_loss(coherent, *args)[0] == pytest.approx(0.025)
    assert position_loss(cancel, *args)[0] == pytest.approx(0.005)


def test_opening_outside_bounds_keeps_corrective_gradient_and_zero_at_agreement():
    loss, grad = opening_loss(np.array([-0.2, 1.2, 0.5]), np.array([0, 1, 0.5]), np.ones(3))
    assert loss == pytest.approx(0.4 / 3)
    np.testing.assert_allclose(grad, [-1 / 3, 1 / 3, 0])


def test_reflection_is_not_a_rotation():
    with pytest.raises(ValueError, match="SO"):
        position_loss(np.zeros((1, 3)), np.diag([-1, 1, 1])[None], np.zeros(3), np.zeros((1, 3)), np.ones(1))
