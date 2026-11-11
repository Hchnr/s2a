"""Keep all cases, generated target cursors and denominators fixed."""

import json

import numpy as np
import pytest

from scripts.tracking.response_models.multicase_slide_table_collision import (
    CONFIG,
    assess_chunk,
    case_summary,
    validate_config,
)


def test_registered_twelve_case_budget_and_no_physical_work():
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    for key, value in (
        ("case_indices", [0]),
        ("targets_per_query", 1),
        ("max_forward_queries", 901),
        ("physics_steps", 1),
        ("world_model_calls", 1),
        ("automatic_retry", True),
    ):
        changed = dict(config)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_config(changed)


def test_all_eight_saved_targets_are_checked_with_distinct_conditions():
    config = json.loads(CONFIG.read_text())
    targets = np.repeat(np.eye(4)[None, :, :], 8, axis=0)
    rows = assess_chunk(targets, [[0, 0, 0]], [1, 0, 0], -0.03, [-0.04, 0], np.eye(4), [0.5, 0.6, 0.05], config)
    assert [row["cursor"] for row in rows] == list(range(8))
    assert all(row["full_joint_range"]["certified"] and row["fixed_current"]["certified"] for row in rows)
    with pytest.raises(ValueError, match="Exactly eight"):
        assess_chunk(targets[:7], [[0, 0, 0]], [1, 0, 0], -0.03, [-0.04, 0], np.eye(4), [0.5, 0.6, 0.05], config)


def test_case_summary_counts_targets_and_queries_separately():
    no = {"fixed_current": {"certified": False}, "full_joint_range": {"certified": False}}
    yes = {"fixed_current": {"certified": True}, "full_joint_range": {"certified": True}}
    queries = [{"targets": [yes] + [no] * 7, "measured_joint_in_range": index != 0} for index in range(75)]
    result = case_summary({"analysis": {"case": {"case_id": "one"}, "queries": queries}})
    assert result["targets"] == 600 and result["queries"] == 75
    assert result["full_joint_range_certified"] == result["queries_with_full_range_certificate"] == 75
    assert result["queries_current_joint_outside_range"] == 1 and result["full_range_by_cursor"] == [
        75,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
