"""Frozen step scales, original constraint admission and complete-suite costs."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from scripts.tracking.trajectory_search.probe_rotation_direction_locality import build_cases, validate_spec
from scripts.tracking.trajectory_search.response_probe_runtime import ResponseProbeRuntime


@pytest.fixture
def spec():
    path = (
        Path(__file__).resolve().parents[3]
        / "configs/tracking/trajectory_search/rotation_direction_locality_20261102.json"
    )
    return json.loads(path.read_bytes())


def data():
    x = np.zeros((12, 6))
    d = x.copy()
    d[-1, 0] = 0.005
    return x, d, np.zeros(6), -np.ones(6), np.ones(6), -np.ones((12, 6)) * 0.1, np.ones((12, 6)) * 0.1


def test_exact_cases_and_two_small_nominal_scales(spec):
    x, d, previous, lo, hi, lower, upper = data()
    cases = build_cases(x, d, spec, previous, lo, hi, lower, upper)
    assert len(cases) == 8
    assert [c["role"] for c in cases] == ["anchor"] + ["backtracking"] * 4 + ["small_step"] * 2 + ["anchor_repeat"]
    assert_allclose([c["alpha"] for c in cases], [0, 0.5, 0.25, 0.125, 0.0625, 0.02, 0.01, 0], atol=1e-17, rtol=0)
    assert_allclose([c["actual_step_inf"] for c in cases[5:7]], [1e-4, 5e-5], atol=2e-17, rtol=0)
    for case in cases:
        assert_array_equal(case["commands"], x + case["alpha"] * d)
    assert_array_equal(x, np.zeros((12, 6)))


@pytest.mark.parametrize("fault", ["global", "trust", "small_direction"])
def test_failed_admission_is_not_clipped_or_resampled(spec, fault):
    x, d, previous, lo, hi, lower, upper = data()
    if fault == "global":
        upper[-1, 0] = 0.0009
    elif fault == "trust":
        d *= 10
    else:
        d *= 0.01
    before = d.copy()
    with pytest.raises(ValueError):
        build_cases(x, d, spec, previous, lo, hi, lower, upper)
    assert_array_equal(d, before)


@pytest.mark.parametrize("fault", ["alphas", "scales", "budget", "retry"])
def test_fixed_specification_rejects_changes(spec, fault):
    if fault == "alphas":
        spec["backtracking_alphas"].reverse()
    elif fault == "scales":
        spec["small_step_inf"][0] *= 2
    elif fault == "budget":
        spec["total_response_budget"] += 2
    else:
        spec["automatic_retry"] = True
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_runtime_rejects_partial_horizon_before_context_or_model_access():
    runtime = ResponseProbeRuntime.__new__(ResponseProbeRuntime)
    cases = [dict(role="anchor", commands=np.zeros((11, 6))), dict(role="anchor_repeat", commands=np.zeros((11, 6)))]
    with pytest.raises(ValueError, match="H12"):
        runtime.run(cases, None, None, None, [], dict(response_count_per_side=2))


def test_runtime_rejects_underreported_costs_before_context_or_model_access():
    runtime = ResponseProbeRuntime.__new__(ResponseProbeRuntime)
    cases = [dict(role="anchor", commands=np.zeros((12, 6))), dict(role="anchor_repeat", commands=np.zeros((12, 6)))]
    with pytest.raises(ValueError, match="Budgets"):
        runtime.run(
            cases,
            None,
            None,
            None,
            [],
            dict(response_count_per_side=2, total_query_budget=1248, total_inner_step_budget=1200),
        )
