"""Keep the recorded cohort identity and validate each selection receipt."""

import copy
import json

import numpy as np
import pytest

from s2a.research.joint_control.preview.nearest_bounded_seed import choose_nearest_bounded_seed
from scripts.simulation.runtime import run_cached_seed_preference as base
from scripts.simulation.runtime.run_nearest_bounded_seed import (
    CONFIG,
    OUTPUT,
    configured,
    summarize_seed_traces,
    validate_config,
)
from tests.research.joint_control.preview.test_cached_seed_preference import candidate


def test_trace_accepts_bounded_intermediate_and_rejects_changed_invariants():
    seed = np.full((8, 6), 0.1)
    candidates = [candidate(0.0, 0.1), candidate(0.06, 0.25), candidate(0.1, 0.5)]
    _chosen, trace = choose_nearest_bounded_seed(candidates, seed)
    summary = summarize_seed_traces([trace])
    assert summary["selected_changed_from_bounded"] == summary["blocked_by_future_slack"] == 1
    assert summary["maximum_future_slack_increase"] == pytest.approx(0.15)
    for field, value in (
        ("future_slack_increase", 0.31),
        ("extra_model_responses", 1),
        ("selected_seed_distance_squared", trace["bounded_seed_distance_squared"] + 1),
        ("bounded_candidates", 0),
    ):
        with pytest.raises(ValueError):
            summarize_seed_traces([dict(trace, **{field: value})])


def test_trace_accepts_empty_original_fallback_without_inventing_a_candidate():
    missing = dict(
        seed_in_eligible_cache=False,
        selected_seed=False,
        selected_changed=False,
        seed_blocked_by_future_slack=False,
        future_slack_increase=None,
        unbounded_seed_future_slack_increase=None,
        max_future_slack_increase=0.3,
        extra_model_responses=0,
        eligible_candidates=0,
        bounded_candidates=0,
        selected_seed_distance_squared=None,
        bounded_seed_distance_squared=None,
        default_seed_distance_squared=None,
        changed_from_bounded=False,
        distance_metric="squared_euclidean_8x6_normalized_arm_commands",
    )
    summary = summarize_seed_traces([missing])
    assert summary["searches"] == 1 and summary["maximum_future_slack_increase"] is None
    with pytest.raises(ValueError):
        summarize_seed_traces([dict(missing, selected_seed_distance_squared=0.0)])


def test_fixed_config_and_isolated_owner_preserve_prior_experiment():
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    for key, value in (("max_future_slack_increase", 1.0), ("joint_buffer_rad", 0.01), ("automatic_retry", 1)):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_config(changed)
    old_output = base.phase_run.__globals__["OUTPUT"]
    isolated = configured(base.phase_run)
    assert isolated.__globals__["OUTPUT"] == OUTPUT
    assert base.phase_run.__globals__["OUTPUT"] == old_output != OUTPUT
