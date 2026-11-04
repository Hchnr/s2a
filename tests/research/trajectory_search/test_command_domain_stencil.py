"""Admissible coupled geometry, identifiability and causal model reconstruction."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from s2a.research.trajectory_search.command_domain_stencil import (
    STEP,
    fit_causal_response,
    prefix_geometry,
    select_interval_basis,
)


def test_saturated_rate_chain_requires_coupled_probes_but_preserves_full_rank():
    x = np.zeros((12, 6))
    x[:, 0] = 0.75
    x[0, 0] = 0.25
    x[1, 0] = 0.5
    basis = select_interval_basis(x, np.zeros(6), -np.ones(6) * 2, np.ones(6) * 2, x - 0.1, x + 0.1)
    assert len(basis["chosen"]) == 72
    assert any(c["length"] > 1 for c in basis["chosen"])
    assert basis["condition"] < 10
    for case in basis["chosen"]:
        assert np.max(abs(np.diff(np.vstack((np.zeros(6), case["commands"])), axis=0))) <= 0.25 + 1e-12
        assert_array_equal(case["actual_displacement"], (case["commands"] - x).ravel())
    assert_array_equal(x[:3, 0], [0.25, 0.5, 0.75])


def test_zero_width_domain_is_not_silently_reduced_or_relaxed():
    x = np.zeros((12, 6))
    with pytest.raises(ValueError, match="complete admissible"):
        select_interval_basis(x, np.zeros(6), -np.ones(6), np.ones(6), x, x)


@pytest.mark.parametrize("fault", ["rank", "condition"])
def test_unidentifiable_design_rejected(fault):
    D = np.eye(72) * STEP
    D[-1, -1] = 0 if fault == "rank" else STEP / 100
    with pytest.raises(ValueError, match="causal prefix"):
        prefix_geometry(D)


def test_known_causal_linear_responses_recovered_from_coupled_design():
    D = np.eye(72) * STEP
    D[0, 6] = STEP
    periods = np.array([0, 1, 5, 11])
    rng = np.random.default_rng(42)
    J = rng.normal(size=(4, 72))
    for i, p in enumerate(periods):
        J[i, 6 * (p + 1) :] = 0
    changes = D @ J.T
    fitted = fit_causal_response(D, changes, periods)
    assert_allclose(fitted["jacobian"], J, atol=1e-14, rtol=0)
    for i, p in enumerate(periods):
        assert_array_equal(fitted["jacobian"][i, 6 * (p + 1) :], 0)
    assert np.max(abs(fitted["residual"])) < 1e-17


def test_future_inconsistent_observation_is_retained_as_residual_not_future_parameter():
    D = np.eye(72) * STEP
    changes = np.zeros((72, 1))
    changes[-1, 0] = 1
    fitted = fit_causal_response(D, changes, [0])
    assert_array_equal(fitted["jacobian"], np.zeros((1, 72)))
    assert fitted["maximum_absolute_residual_by_row"][0] == 1
