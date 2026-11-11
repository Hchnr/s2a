"""Physical admission must preserve reference identity and declared limits."""

import copy
import json

import pytest

from scripts.simulation.runtime.prepare_box_face_feedback import CONFIG, validate_config


def test_registered_physical_budget_is_admitted():
    config = json.loads(CONFIG.read_text())
    validate_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("new_period_limit", 8),
        ("max_total_native_steps", 6400),
        ("new_model_requests", 1),
        ("mode", "pose_bounded"),
        ("max_translation_m", 0.06),
        ("command_step_limit", 0.5),
        ("automatic_retry", True),
    ],
)
def test_budget_or_reference_contract_cannot_be_silently_changed(key, value):
    config = copy.deepcopy(json.loads(CONFIG.read_text()))
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)
