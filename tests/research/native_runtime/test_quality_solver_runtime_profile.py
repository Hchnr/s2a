"""Profiling must preserve solver results and finite response accounting."""

import copy
import cProfile

import numpy as np
import pytest

from scripts.simulation.runtime import profile_quality_solver_runtime as profile


def test_profile_rows_report_measured_calls_and_times():
    def sample_work():
        return sum(range(50))

    profiler = cProfile.Profile()
    profiler.runcall(sample_work)
    rows = profile.profile_rows(profiler)
    item = next(row for row in rows if row["function"] == "sample_work")
    assert item["primitive_calls"] == item["total_calls"] == 1
    assert item["self_seconds"] >= 0 and item["cumulative_seconds"] >= item["self_seconds"]


def test_profile_runs_same_inputs_twice_without_auditing_a_nonwitness():
    sample = {
        "step": 352,
        "mass": np.eye(7),
        "jacobian": np.zeros((6, 7)),
        "kp": np.ones(6),
        "kd": np.ones(6),
        "q": np.zeros(7),
        "initial_joint": np.zeros(7),
        "velocity": np.zeros(7),
        "seed": np.ones((8, 6)) / 10,
        "previous": np.zeros(6),
        "site": np.eye(4),
        "site_from_tcp": np.eye(4),
        "targets": np.tile(np.eye(4), (8, 1, 1)),
        "twists": np.zeros((8, 6)),
        "lower": np.full(7, -2.0),
        "upper": np.full(7, 2.0),
        "past_inner_error_rad": np.zeros((25, 7)),
        "contract": {"input_min": [-1] * 6, "input_max": [1] * 6, "output_min": [-1] * 6, "output_max": [1] * 6},
    }
    config = {"null_kp": 10.0, "inner_dt_s": 0.002, "max_response_evaluations_per_call": 160}
    ledger = dict.fromkeys(
        (
            "solver_calls_attempted",
            "solver_calls_completed",
            "response_evaluations_completed",
            "independent_audits_attempted",
            "independent_audits_completed",
        ),
        0,
    )
    calls = []

    def solver(*args, **kwargs):
        calls.append((copy.deepcopy(args[2]), copy.deepcopy(kwargs["past_inner_error_rad"])))
        return {
            "status": "nonlinear_search_no_witness",
            "witness_found": False,
            "response_evaluations": 7,
            "optimizer_success": False,
        }

    def forbidden_audit(*_args, **_kwargs):
        raise AssertionError("No witness must not be audited")

    result = profile.run_sample(
        sample, config, ledger, builder=lambda *_a, **_k: {}, solver=solver, auditor=forbidden_audit
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    assert ledger["solver_calls_attempted"] == ledger["solver_calls_completed"] == 2
    assert ledger["response_evaluations_completed"] == 14 and ledger["independent_audits_attempted"] == 0
    assert result["witness_found"] is False and result["repeated_result_max_difference"] == 0
    assert result["independent_audit"] is None


def test_repeated_solver_result_differences_are_rejected():
    with pytest.raises(ValueError):
        profile.compare_recorded({"status": "witness"}, {"status": "no_witness"})
