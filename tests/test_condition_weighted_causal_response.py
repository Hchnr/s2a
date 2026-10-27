import numpy as np

from s2a.condition_weighted_causal_response import ConditionWeightedCausalResponse


def test_completed_history_copy_query_purity_and_remote_shrinkage():
    estimator = ConditionWeightedCausalResponse(np.ones(89))
    raw = np.zeros(89)
    prior = np.zeros(12)
    action = np.zeros(3)
    np.testing.assert_array_equal(estimator.predict(raw, action, prior), prior)
    realized = np.zeros(12)
    realized[0] = 0.02
    estimator.observe(raw, action, prior, realized)
    realized[0] = 100
    near = estimator.predict(raw, action, prior)
    np.testing.assert_allclose(near[0], 0.01, atol=1e-14)
    far = estimator.predict(np.ones(89) * 100, action, prior)
    assert abs(far[0]) < 1e-8
    np.testing.assert_array_equal(estimator.predict(raw, action, prior), near)
    assert estimator.updates == 1 and len(estimator.history) == 1
    isolated = ConditionWeightedCausalResponse(np.ones(89))
    np.testing.assert_array_equal(isolated.predict(raw, action, prior), prior)


def test_independent_augmented_weighted_solve_and_window_expiry():
    rng = np.random.default_rng(42)
    estimator = ConditionWeightedCausalResponse(np.ones(89))
    for _ in range(23):
        r = rng.normal(size=89)
        a = rng.normal(size=3)
        y = rng.normal(size=12) * 0.01
        y[3:6] = 0
        estimator.observe(r, a, np.zeros(12), y)
    assert len(estimator.history) == 20
    query = np.zeros(89)
    x = np.stack([r[1] for r in estimator.history])
    y = np.stack([r[2] for r in estimator.history])
    blocks = [
        list(range(7)) + list(range(21, 28)),
        list(range(28, 37)),
        list(range(37, 43)),
        list(range(43, 51)),
        list(range(51, 71)),
        list(range(71, 83)),
        list(range(83, 89)),
    ]
    weights = []
    for raw, _, _ in estimator.history:
        d = sum(sum(raw[k] ** 2 for k in b) / len(b) for b in blocks) / 7
        weights.append((1 + d) ** -2 / 20)
    root = np.sqrt(weights)[:, None]
    independent = np.linalg.lstsq(
        np.vstack([root * x, np.eye(4)]), np.vstack([root * y, np.zeros((4, 12))]), rcond=None
    )[0]
    np.testing.assert_allclose(estimator.coefficients(query), independent, rtol=1e-12, atol=1e-14)
