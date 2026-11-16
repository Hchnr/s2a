"""Protect the seed-first physical comparison's provenance and accounting."""

import copy
import json

import pytest

from scripts.simulation.runtime import run_fast_seed_cohort as run


def test_fixed_config_cannot_change_a_safety_gate_or_source():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for key, value in (("joint_buffer_rad", 0.01), ("gpu_calls", 1), ("candidate_cases", 17)):
        with pytest.raises(ValueError):
            run.validate_config(dict(config, **{key: value}))
    changed = copy.deepcopy(config)
    changed["fixed_state_comparison"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        run.validate_config(changed)


def test_seed_trace_requires_both_guards_and_unknown_comparison():
    early = dict(
        early_selected=True,
        original_solver_continued=False,
        seed_domain_accepted=True,
        seed_margin_accepted=True,
        baseline_future_slack=None,
        selected_future_slack=0.1,
        future_slack_increase=None,
        additional_model_responses=0,
    )
    fallback = dict(
        early_selected=False,
        original_solver_continued=True,
        seed_probe_reached=True,
        seed_domain_accepted=False,
        seed_margin_accepted=True,
        additional_model_responses=0,
    )
    summary = run.summarize_seed_traces([early, fallback])
    assert summary["early_selected"] == summary["original_solver_continued"] == 1
    for key, value in (
        ("seed_margin_accepted", False),
        ("future_slack_increase", 0),
        ("additional_model_responses", 1),
    ):
        with pytest.raises(ValueError):
            run.summarize_seed_traces([dict(early, **{key: value})])
    with pytest.raises(ValueError):
        run.summarize_seed_traces([dict(fallback, seed_domain_accepted=True)])


def test_policy_timing_covers_the_whole_select_and_preserves_result(monkeypatch):
    class Policy:
        def select(self, value):
            return {"command": [value], "evidence": "retained"}

    policy = Policy()
    monkeypatch.setattr(Policy, "select", Policy.select)
    run.instrument_policy_selection(policy)
    result = policy.select(4)
    assert result["command"] == [4] and result["evidence"] == "retained"
    assert result["full_policy_select_seconds"] >= 0


def test_prior_comparison_detects_changed_action_and_short_execution():
    prior = run.prefix.read(run.preparation.configured_ref(run.UNBOUNDED_AUDIT))
    branch = run.prefix.read(prior["cases"][0]["producer_ref"])
    same = run.compare_prior_execution(branch, branch)
    assert same["same_complete_execution"] is True and same["first_numeric_difference"] is None
    changed = copy.deepcopy(branch)
    run.previous.periods(changed)[0]["command"][0] += 0.001
    diff = run.compare_prior_execution(changed, branch)
    assert diff["same_complete_execution"] is False and diff["first_numeric_difference"] == 0
    changed = copy.deepcopy(branch)
    changed["runtime"]["continuation"]["segments"][-1]["continuation"]["periods"].pop()
    assert run.compare_prior_execution(changed, branch)["same_complete_execution"] is False
