"""Keep fixed-current and full nominal-range collision claims distinct."""

import json

import numpy as np
import pytest

from scripts.tracking.response_models.slide_range_table_collision import CONFIG, assess_target, validate_config


def test_saved_target_reports_both_conditional_intervals():
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    decision = {"original_target_tcp": np.eye(4), "decision_vertices_tcp": [[0, 0, 0]]}
    result = assess_target(decision, [1, 0, 0], -0.041, [-0.04, 0], np.eye(4), [0.5, 0.6, 0.05], config)
    assert result["fixed_current"]["certified"] and result["full_joint_range"]["certified"]
    assert result["fixed_current"]["joint_target_interval_m"] == [-0.041, -0.041]
    assert result["full_joint_range"]["joint_target_interval_m"] == [-0.04, 0.0]
    assert not result["full_joint_range"]["measured_joint_in_target_interval"]


def test_analysis_rejects_new_physics_or_widened_tolerance():
    config = json.loads(CONFIG.read_text())
    for key, value in (("physics_steps", 1), ("position_tolerance_m", 0.02), ("max_early_endpoints", 15)):
        modified = dict(config)
        modified[key] = value
        with pytest.raises(ValueError):
            validate_config(modified)
