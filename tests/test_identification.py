import numpy as np
from numpy.testing import assert_allclose

from s2a.identification import LinearResponse, history_vector, replay_model, temporal_support


def test_recovers_coupled_gain_bias_and_regularized_inverse_without_test_fit():
    rng = np.random.default_rng(42)
    u = rng.normal(size=(121, 6)) * 0.1
    b = np.diag(np.arange(1, 7) * 0.1)
    b[0, 1] = 0.03
    bias = np.array([0.01, 0, 0, 0, 0, 0])
    y = u @ b.T + bias
    ids = temporal_support(np.arange(121), [0, 100], 0, 0)
    model = LinearResponse.fit(u, y, ids, ridge=1e-9)
    assert_allclose(model.B, b, atol=1e-8)
    assert_allclose(model.bias, bias, atol=1e-8)
    wanted = y[110]
    result = model.inverse(wanted, np.empty(0), u[110], 0.1)
    assert_allclose(model.predict(result, np.empty(0)), wanted, atol=1e-9)
    y[100:] = 1000
    again = LinearResponse.fit(u, y, ids, ridge=1e-9)
    assert_allclose(again.B, model.B)


def test_delayed_replay_queue_and_past_motion_history_are_causal():
    model = LinearResponse(
        np.eye(6), np.c_[np.zeros((6, 6)), np.eye(6) * 0.5], np.zeros(6), 2, 1, "full", 0.1, np.ones(6), {}
    )
    u = np.zeros((8, 6))
    u[0, 0] = 1
    y = replay_model(model, u)
    assert_allclose(y[:2], 0)
    assert y[2, 0] == 1
    assert y[5, 0] == 0.5
    before = history_vector(u, y, 3, 1)
    y[3:] = 999
    u[3:] = 999
    assert_allclose(history_vector(u, y, 3, 1), before)
