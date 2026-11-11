"""Different starts may be compared only on the same causal safety domain."""

import copy

import numpy as np
import pytest

from scripts.tracking.trajectory_search import compare_task2_shifted_seed_search as runner


def test_domain_comparison_rejects_changed_rows_or_terminal_direction():
    keys = (
        "lo",
        "hi",
        "rate",
        "center",
        "joint_a",
        "joint_lo",
        "joint_hi",
        "terminal_a",
        "terminal_offset",
        "terminal_normals",
    )
    first = {key: np.ones((2, 2)) for key in keys}
    second = copy.deepcopy(first)
    assert all(value == 0 for value in runner.compare_domains(first, second).values())
    second["joint_a"][0, 0] += 1e-5
    with pytest.raises(ValueError, match="same affine safety domain"):
        runner.compare_domains(first, second)
    second = copy.deepcopy(first)
    second["terminal_normals"] = np.empty((0, 2))
    with pytest.raises(ValueError, match="same affine safety domain"):
        runner.compare_domains(first, second)


def test_saved_lp_starts_are_bound_to_the_two_replayed_stops():
    inputs = runner.load_inputs()
    assert [item["stop_spec"] for item in inputs["cases"]] == runner.CASES
    for case in inputs["cases"]:
        assert case["saved_lp"]["commands"].shape == (8, 6)
        assert case["saved_lp"]["prior_response_evaluations"] <= 320
        assert not case["shifted_qualification"]["model_prefix_candidate"]
    assert runner.input_receipt(inputs)["prior_ref"] == runner.PRIOR


def test_search_accounting_rejects_uncharged_independent_work():
    stats = runner.new_stats()
    stats.update(
        model_initializations_attempted=1,
        model_initializations_completed=1,
        forward_queries_attempted=1,
        forward_queries_completed=1,
        baseline_rollouts=1,
        additional_preparation_rollouts=1,
        solver_calls_attempted=2,
        solver_calls_completed=2,
        solver_responses_attempted=25,
        solver_responses_completed=25,
        independent_audits_attempted=1,
        independent_audits_passed=1,
        independent_endpoint_queries_charged=200,
        independent_transition_steps_charged=2600,
    )
    runner.require_case_counts(stats)
    stats["independent_transition_steps_charged"] -= 1
    with pytest.raises(ValueError, match="costs"):
        runner.require_case_counts(stats)
