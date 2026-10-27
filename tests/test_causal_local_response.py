import numpy as np
from scipy.spatial.transform import Rotation

from s2a.causal_local_response import CausalLocalResponse


def test_reading_prediction_does_not_update_and_episodes_are_isolated():
    a, b = CausalLocalResponse(), CausalLocalResponse()
    prior = np.zeros(12)
    a.observe(np.zeros(3), prior, np.ones(12) * 0.1)
    first = a.predict(np.zeros(3), prior)
    np.testing.assert_array_equal(a.predict(np.zeros(3), prior), first)
    assert a.updates == 1 and b.updates == 0
    np.testing.assert_array_equal(b.predict(np.zeros(3), prior), prior)


def test_coefficients_match_independent_augmented_svd_and_window():
    rng = np.random.default_rng(42)
    model = CausalLocalResponse()
    history = []
    for _ in range(30):
        action, target = rng.normal(size=3), rng.normal(size=12) * 0.01
        model.observe(action, np.zeros(12), target)
        history.append((np.r_[1, action], target))
    x = np.stack([row[0] for row in history[-20:]]) / np.sqrt(20)
    y = np.stack([row[1] for row in history[-20:]]) / np.sqrt(20)
    beta = np.linalg.lstsq(np.vstack([x, np.eye(4)]), np.vstack([y, np.zeros((4, 12))]), rcond=None)[0]
    np.testing.assert_allclose(model.coefficients(), beta, rtol=0, atol=1e-14)
    assert len(model.history) == 20


def test_noncommuting_world_rotation_innovation_and_composition():
    prior = np.zeros(12)
    prior[3:6] = [0.6, 0.2, -0.1]
    innovation = np.array([0.1, -0.2, 0.3])
    actual = prior.copy()
    actual[3:6] = (Rotation.from_rotvec(innovation) * Rotation.from_rotvec(prior[3:6])).as_rotvec()
    model = CausalLocalResponse()
    model.observe(np.zeros(3), prior, actual)
    expected = Rotation.from_rotvec(innovation / 2) * Rotation.from_rotvec(prior[3:6])
    np.testing.assert_allclose(
        Rotation.from_rotvec(model.predict(np.zeros(3), prior)[3:6]).as_matrix(), expected.as_matrix(), atol=1e-14
    )


def test_future_realizations_do_not_change_earlier_prequential_predictions():
    def replay(realizations):
        model = CausalLocalResponse()
        predictions = []
        for i in range(len(realizations)):
            if i:
                model.observe(np.zeros(3), np.zeros(12), realizations[i - 1])
            predictions.append(model.predict(np.zeros(3), np.zeros(12)))
        return np.array(predictions)

    first = np.zeros((8, 12))
    second = first.copy()
    second[4:, :3] = 10
    np.testing.assert_array_equal(replay(first)[:5], replay(second)[:5])
    assert not np.array_equal(replay(first)[5:], replay(second)[5:])
