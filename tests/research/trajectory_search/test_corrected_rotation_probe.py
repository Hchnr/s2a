"""Safety fit must preserve causality and reject inadmissible observations."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from scripts.tracking.trajectory_search.probe_corrected_rotation import correct_safety


def fixture():
    config = dict(
        safety_buffer=1e-8, secant_min_squared_step=1e-12, secant_relative_limit=2.0, secant_row_norm_floor=1e-6
    )
    anchor = dict(commands=np.zeros((12, 6)), safety_slack=np.ones(4369), audit=dict(feasible=True))
    fit = copy.deepcopy(anchor)
    fit["commands"][-1, 0] = 1e-4
    # Earlier response rows cannot depend on this last-period step.
    fit["safety_slack"][-1] += 1e-10
    return np.zeros((4369, 72)), anchor, fit, config


def test_last_period_update_has_no_effect_on_earlier_rows_or_inputs():
    J, anchor, fit, config = fixture()
    updated, record = correct_safety(J, anchor, fit, config)
    assert_array_equal(updated[:-1], J[:-1])
    assert updated[-1, 66] > 0
    assert np.count_nonzero(updated) == 1
    assert record["theta"][-1] == 1
    assert_array_equal(J, np.zeros_like(J))
    assert_array_equal(anchor["commands"], np.zeros((12, 6)))


@pytest.mark.parametrize("fault", ["unsafe", "buffer", "large", "zero", "partial", "nan"])
def test_invalid_fit_is_rejected(fault):
    J, anchor, fit, config = fixture()
    if fault == "unsafe":
        fit["audit"]["feasible"] = False
    elif fault == "buffer":
        fit["safety_slack"][0] = 0
    elif fault == "large":
        fit["commands"][-1, 0] = 0.001
    elif fault == "zero":
        fit["commands"][:] = 0
    elif fault == "partial":
        fit["commands"] = np.zeros((11, 6))
    else:
        fit["commands"][-1, 0] = np.nan
    with pytest.raises(ValueError):
        correct_safety(J, anchor, fit, config)
