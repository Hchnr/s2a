"""The current-image pilot must retain its registered task and budgets."""

import json

import pytest

from scripts.simulation.runtime import reset_live_second_admission as admission


def test_live_registration_rejects_changes_to_task_device_and_budgets():
    config = json.loads(admission.CONFIG.read_text())
    admission.validate_config(config)
    for key, value in (
        ("gpu_uuid", "0"),
        ("case_index", 0),
        ("task", "different task"),
        ("max_model_requests", 2),
        ("max_native_steps_per_phase", 401),
        ("fresh_cases", False),
        ("automatic_retry", True),
    ):
        with pytest.raises(ValueError, match="exact reset"):
            admission.validate_config(dict(config, **{key: value}))
