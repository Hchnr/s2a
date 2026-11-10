"""The benchmark must compare full outputs and account for every call."""

import numpy as np
import pytest

from scripts.simulation.runtime import benchmark_batched_joint_rollout as benchmark
from tests.research.joint_control.preview.test_batched_free_joint_rollout import example


def test_output_comparison_rejects_changed_jacobians_and_discrete_labels():
    expected = {"jacobian": np.eye(3), "labels": np.array([0, 1])}
    actual = {key: value.copy() for key, value in expected.items()}
    assert max(benchmark.compare_outputs(expected, actual).values()) == 0
    actual["jacobian"][0, 0] += 1e-4
    with pytest.raises(AssertionError):
        benchmark.compare_outputs(expected, actual)
    actual = {key: value.copy() for key, value in expected.items()}
    actual["labels"][1] = 0
    with pytest.raises(AssertionError):
        benchmark.compare_outputs(expected, actual)


def test_benchmark_alternates_and_accounts_for_warmup_and_timed_pairs():
    model, velocity, deltas, site, lower, upper = example(4)
    sample = {
        "step": 352,
        "mass": model["mass"],
        "jacobian": model["jacobian"],
        "kp": np.ones(6) * 100,
        "kd": np.ones(6) * 20,
        "q": model["q"],
        "initial_joint": np.zeros(7),
        "velocity": velocity,
        "seed": deltas,
        "site": site,
        "lower": lower,
        "upper": upper,
        "contract": {"input_min": [-1] * 6, "input_max": [1] * 6, "output_min": [-1] * 6, "output_max": [1] * 6},
    }
    config = {
        "null_kp": 10.0,
        "inner_dt_s": 0.002,
        "comparison_atol": 1e-10,
        "warmup_calls_per_implementation": 1,
        "timed_calls_per_implementation": 2,
    }
    ledger = dict.fromkeys(
        (
            "sequential_calls_attempted",
            "sequential_calls_completed",
            "batched_calls_attempted",
            "batched_calls_completed",
        ),
        0,
    )
    result = benchmark.benchmark_sample(sample, config, ledger)
    assert all(value == 3 for value in ledger.values())
    assert result["compared_pairs"] == 3
    assert all(len(result["timings"][mode]) == 2 for mode in ("sequential", "batched"))
    assert max(result["max_abs_difference_by_field"].values()) <= 1e-10
