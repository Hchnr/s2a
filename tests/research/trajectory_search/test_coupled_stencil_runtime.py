"""Independent causal fit and observed-prefix invariance are distinct checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from s2a.research.trajectory_search.command_domain_stencil import fit_causal_response
from scripts.tracking.trajectory_search.coupled_stencil_runtime import (
    exact_prefix_observations,
    independent_causal_fit,
)


def test_independent_normal_equations_reconstruct_known_causal_response():
    D = np.eye(72) * 1e-4
    D[0, 6] = 1e-4
    periods = np.repeat(np.arange(12), 2)
    rng = np.random.default_rng(42)
    J = rng.normal(size=(24, 72))
    for i, p in enumerate(periods):
        J[i, 6 * (p + 1) :] = 0
    changes = D @ J.T
    audit = independent_causal_fit(D, changes, periods)
    production = fit_causal_response(D, changes, periods)
    assert_allclose(audit["jacobian"], J, atol=1e-13, rtol=0)
    assert_allclose(production["jacobian"], audit["jacobian"], atol=1e-13, rtol=0)
    for i, p in enumerate(periods):
        assert_array_equal(audit["jacobian"][i, 6 * (p + 1) :], 0)


def test_identical_future_only_command_changes_have_exact_zero_past_output():
    D = np.eye(72) * 1e-4
    Y = np.zeros((72, 1))
    Y[:6, 0] = np.arange(6)
    record = exact_prefix_observations(D, Y, [0])
    assert record["exact"] and record["zero_prefix_checks"] > 0
    Y[-1, 0] = 1e-16
    with pytest.raises(ValueError, match="Zero observed prefix"):
        exact_prefix_observations(D, Y, [0])


def test_identical_nonzero_prefix_outputs_must_match_exactly():
    D = np.eye(72) * 1e-4
    D[-1, 0] = 1e-4
    Y = np.zeros((72, 1))
    Y[0, 0] = 1
    Y[-1, 0] = 1 + 1e-15
    with pytest.raises(ValueError, match="Same observed prefix"):
        exact_prefix_observations(D, Y, [0])


@pytest.mark.parametrize("fault", ["rank", "condition", "nonfinite", "period"])
def test_independent_fit_rejects_invalid_inputs(fault):
    D = np.eye(72) * 1e-4
    Y = np.zeros((72, 1))
    periods = [11]
    if fault == "rank":
        D[-1, -1] = 0
    elif fault == "condition":
        D[-1, -1] /= 100
    elif fault == "nonfinite":
        Y[-1, 0] = np.nan
    else:
        periods = [12]
    with pytest.raises(ValueError):
        independent_causal_fit(D, Y, periods)
