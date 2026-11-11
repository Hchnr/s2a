"""The early comparison keeps the same reference, modes and native budget."""

import json

import pytest

from scripts.simulation.runtime.prepare_early_box_face_feedback import CONFIG, validate_config


def test_early_comparison_accepts_registered_controls():
    validate_config(json.loads(CONFIG.read_text()))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("branch_step", 120),
        ("reference_step", 120),
        ("modes", ["box_face"]),
        ("max_total_native_steps", 13000),
        ("new_model_requests", 1),
        ("new_period_limit", 6),
        ("automatic_retry", True),
    ],
)
def test_changed_early_reference_or_budget_is_rejected(key, value):
    config = json.loads(CONFIG.read_text())
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)
