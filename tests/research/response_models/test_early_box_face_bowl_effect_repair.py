"""The repair must retain prior spent work and strict normalization bounds."""

import json

import pytest

from scripts.tracking.response_models.early_box_face_bowl_effect_repair import CONFIG, validate_config


def test_repair_accepts_only_registered_cumulative_budget():
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    for key, value in (
        ("prior_forward_queries", 0),
        ("total_max_forward_queries", 13),
        ("quaternion_atol", 1e-6),
        ("physics_steps", 1),
        ("automatic_retry", True),
    ):
        modified = dict(config)
        modified[key] = value
        with pytest.raises(ValueError):
            validate_config(modified)
