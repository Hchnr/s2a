"""The bounded cohort retains the registered gates and trace accounting."""

import copy
import json

import pytest

from scripts.simulation.runtime import run_cached_seed_preference as base
from scripts.simulation.runtime.run_bounded_seed_preference import (
    CONFIG,
    MAX_INCREASE,
    OUTPUT,
    configured,
    summarize_seed_traces,
    validate_config,
)


def trace(**changes):
    return dict(
        seed_in_eligible_cache=True,
        selected_seed=False,
        selected_changed=False,
        seed_blocked_by_future_slack=True,
        future_slack_increase=0.0,
        unbounded_seed_future_slack_increase=0.4,
        max_future_slack_increase=MAX_INCREASE,
        extra_model_responses=0,
        **changes,
    )


def test_bounded_trace_records_rejection_and_zero_extra_responses():
    row = trace()
    result = summarize_seed_traces([row])
    assert result["blocked_by_future_slack"] == 1
    assert result["maximum_blocked_seed_increase"] == 0.4
    assert result["maximum_future_slack_increase"] == 0.0
    for field, value in (
        ("selected_seed", True),
        ("unbounded_seed_future_slack_increase", 0.2),
        ("extra_model_responses", 1),
        ("future_slack_increase", 0.1),
    ):
        changed = dict(row, **{field: value})
        with pytest.raises(ValueError):
            summarize_seed_traces([changed])


def test_bounded_trace_accepts_valid_seed_and_no_cache_fallback():
    accepted = dict(
        seed_in_eligible_cache=True,
        selected_seed=True,
        selected_changed=True,
        seed_blocked_by_future_slack=False,
        future_slack_increase=0.3,
        unbounded_seed_future_slack_increase=0.3,
        max_future_slack_increase=MAX_INCREASE,
        extra_model_responses=0,
    )
    missing = dict(
        seed_in_eligible_cache=False,
        selected_seed=False,
        selected_changed=False,
        seed_blocked_by_future_slack=False,
        future_slack_increase=None,
        unbounded_seed_future_slack_increase=None,
        max_future_slack_increase=MAX_INCREASE,
        extra_model_responses=0,
    )
    result = summarize_seed_traces([accepted, missing])
    assert result["searches"] == 2 and result["selected_changed"] == 1
    assert result["maximum_future_slack_increase"] == 0.3


def test_registered_config_rejects_bound_or_safety_changes():
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    for key, value in (("max_future_slack_increase", 1.0), ("joint_buffer_rad", 0.01), ("automatic_retry", 1)):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_config(changed)


def test_configured_cohort_owner_does_not_mutate_the_previous_experiment():
    old_output = base.phase_run.__globals__["OUTPUT"]
    isolated = configured(base.phase_run)
    assert isolated.__globals__["OUTPUT"] == OUTPUT
    assert base.phase_run.__globals__["OUTPUT"] == old_output != OUTPUT
