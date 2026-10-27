import numpy as np

from s2a.reference_temporal_error import temporal_terms


def test_coherent_error_has_positive_temporal_cross_term():
    result = temporal_terms([[0.1, 0, 0], [0.1, 0, 0]])
    np.testing.assert_allclose(result["local_diagonal_cumulative_m2"], [0.01, 0.02])
    np.testing.assert_allclose(result["local_temporal_cross_m2"], [0, 0.02], atol=1e-15)


def test_reversing_error_cancels_without_changing_diagonal_energy():
    result = temporal_terms([[0.1, 0, 0], [-0.1, 0, 0]])
    np.testing.assert_allclose(result["local_diagonal_cumulative_m2"], [0.01, 0.02])
    np.testing.assert_allclose(result["local_temporal_cross_m2"], [0, -0.02], atol=1e-15)
