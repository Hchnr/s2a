import numpy as np
import pytest

from s2a.diagonal_quadratic_action_forward import expand


def branch(phi, action):
    return np.r_[phi, *(phi * a for a in action)][None, :]


def test_asymmetric_interpolation_cancels_affine_terms_and_recovers_axis_curvature():
    rng = np.random.default_rng(42)
    phi = rng.normal(size=282)
    phi[0] = 1
    action = np.array([0.4, -0.3, 0.2])
    minus = action.copy()
    minus[1] -= 0.08
    plus = action.copy()
    plus[1] += 0.12
    fraction = 0.08 / 0.2
    center, m, p = (expand(branch(phi, a))[0] for a in (action, minus, plus))
    residual = center - (1 - fraction) * m - fraction * p
    np.testing.assert_allclose(residual[:1128], 0, atol=1e-14)
    np.testing.assert_allclose(residual[1128:1410], 0, atol=1e-14)
    np.testing.assert_allclose(residual[1410:1692], -0.08 * 0.12 * phi, atol=1e-14)
    np.testing.assert_allclose(residual[1692:], 0, atol=1e-14)
    # Centered differenced equations cannot be expanded as physical branches.
    with pytest.raises(ValueError):
        expand((branch(phi, plus) - branch(phi, minus)) / 0.2)


def test_independent_scalar_features_and_missing_cross_terms():
    rng = np.random.default_rng(43)
    phi = rng.normal(size=282)
    phi[0] = 1
    action = rng.normal(size=3)
    expected = []
    for multiplier in [1, *action, *(a * a for a in action)]:
        expected.extend(float(v) * multiplier for v in phi)
    np.testing.assert_allclose(expand(branch(phi, action))[0], expected, atol=1e-14)
    assert len(expected) == 1974
