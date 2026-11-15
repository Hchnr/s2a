"""The experimental cohort retains every case and accounts for cached choices."""

import copy
import json

import pytest

from scripts.simulation.runtime.run_cached_seed_preference import (
    CONFIG,
    eligible_checker_cases,
    summarize_seed_traces,
    validate_config,
)


def test_checker_includes_case_eight_and_preserves_failed_producers():
    cases = [dict(spec=dict(index=index)) for index in (0, 8, 9)]
    producer = dict(cases={"0": dict(status="failed"), "8": dict(status="closed"), "9": dict(status="closed")})
    assert eligible_checker_cases(producer, cases) == [8, 9]


def test_seed_trace_rejects_unaccounted_or_unqualified_selection():
    row = dict(
        seed_in_eligible_cache=True,
        selected_seed=True,
        selected_changed=True,
        future_slack_increase=0.12,
        extra_model_responses=0,
    )
    assert summarize_seed_traces([row])["selected_changed"] == 1
    for field, value in (
        ("seed_in_eligible_cache", False),
        ("extra_model_responses", 1),
        ("future_slack_increase", -0.1),
    ):
        changed = dict(row, **{field: value})
        with pytest.raises(ValueError):
            summarize_seed_traces([changed])


def test_variant_config_rejects_a_changed_safety_gate():
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    changed = copy.deepcopy(config)
    changed["joint_buffer_rad"] = 0.01
    with pytest.raises(ValueError):
        validate_config(changed)
