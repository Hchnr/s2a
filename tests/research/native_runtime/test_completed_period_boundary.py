"""Validate a streaming boundary at an arbitrary observed control period."""

import copy

import numpy as np
import pytest

from s2a.research.native_runtime.completed_period_boundary import completed_period_inner


def completed():
    limits = np.tile([-1.0, 1.0], (7, 1))
    samples = [{"time": 5.6 - 0.002 * (24 - index), "q": np.zeros(7), "qvel": np.zeros(7)} for index in range(25)]
    return {
        "step": 106,
        "endpoint_receipt": {"endpoint_step": 107, "sim_time": 5.6},
        "actual_inner": samples,
        "inner_below_buffer_samples": 0,
        "inner_outside_samples": 0,
    }, limits


def test_boundary_preserves_observed_inner_history_without_mutating_input():
    period, limits = completed()
    result = completed_period_inner(period, limits)
    assert result["prefix_step"] == 106 and result["below_buffer_samples"] == result["outside_samples"] == 0
    assert result["minimum_margin_rad"] == 1.0 and len(result["samples"]) == 25
    result["samples"][0]["q"][0] = 0.5
    result["limits"][0, 0] = -2.0
    assert period["actual_inner"][0]["q"][0] == 0.0 and limits[0, 0] == -1.0


@pytest.mark.parametrize("change", ["missing", "time", "endpoint", "nan", "counts", "limits"])
def test_invalid_or_misaligned_inner_history_is_rejected(change):
    period, limits = completed()
    period = copy.deepcopy(period)
    if change == "missing":
        period["actual_inner"].pop()
    elif change == "time":
        period["actual_inner"][8]["time"] += 0.001
    elif change == "endpoint":
        period["endpoint_receipt"]["endpoint_step"] += 1
    elif change == "nan":
        period["actual_inner"][0]["qvel"][0] = np.nan
    elif change == "counts":
        period["inner_below_buffer_samples"] = 1
    else:
        limits[0] = [1.0, -1.0]
    with pytest.raises(ValueError):
        completed_period_inner(period, limits)


def test_actual_buffer_loss_is_reported_with_its_observed_count():
    period, limits = completed()
    period["actual_inner"][-1]["q"][0] = 0.96
    period["inner_below_buffer_samples"] = 1
    result = completed_period_inner(period, limits)
    assert result["below_buffer_samples"] == 1 and result["outside_samples"] == 0
    assert result["minimum_margin_rad"] == pytest.approx(0.04)
