"""The fixed-horizon probe must use the audited bowl case and saved response."""

import copy
import json

import pytest

from scripts.simulation.runtime import reset_bowl_fixed_h40_admission as admission


def test_config_rejects_case_response_and_budget_changes():
    config = json.loads(admission.CONFIG.read_text())
    admission.validate_config(config)
    for field, value in (
        ("case_index", 4),
        ("reference_response_sha256", "a" * 64),
        ("reference_image_sha256", "b" * 64),
        ("generated_horizon", 41),
        ("new_model_requests", 1),
        ("previous_periods", 0),
        ("automatic_retry", True),
    ):
        changed = copy.deepcopy(config)
        changed[field] = value
        with pytest.raises(ValueError, match="fixed bowl case"):
            admission.validate_config(changed)


def test_loads_the_real_previously_audited_current_image_and_h40():
    inputs = admission.load_inputs(verify_assets=False)
    assert inputs["case"]["case"]["case_id"] == "task0_init0_s42_raw"
    assert inputs["expected_capture"]["frame_index"] == 8
    assert inputs["refs"]["response"]["sha256"] == admission.RESPONSE_SHA
    assert inputs["refs"]["image"]["sha256"] == admission.IMAGE_SHA
    assert len(inputs["baseline"]) == 8
    actual, envelope = admission.load_reference(inputs)
    assert actual["status"] == "success" and envelope["step"] == 8 and envelope["denoise_calls"] == 35
