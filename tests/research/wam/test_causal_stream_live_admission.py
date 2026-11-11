"""Check live WAM provenance and budget before any model initialization."""

import json

import pytest

from scripts.wam.tracking.preview import causal_stream_live_admission as admission


def test_prior_physics_and_model_rosters_are_bound_without_launch():
    inputs = admission.load_inputs()
    assert inputs["live_config"]["gpu_uuid"] == admission.GPU
    assert [row["step"] for row in inputs["records"]] == [80, 88]
    assert len(inputs["model_source_hashes"]) >= 100
    assert len(inputs["weight_hashes"]) == 163
    assert inputs["prepared"]["model_config"]["checkpoint"]["name"] == "8400_masked"
    json.dumps(admission.input_receipt(inputs), allow_nan=False)


@pytest.mark.parametrize(
    "key",
    [
        "gpu_uuid",
        "model_initializations",
        "model_requests",
        "max_denoise_calls",
        "max_native_steps_per_phase",
        "automatic_retry",
    ],
)
def test_any_gpu_or_budget_change_is_rejected(key):
    config = dict(admission.FIXED)
    config[key] = "other" if key == "gpu_uuid" else (2 if key == "model_initializations" else 1)
    with pytest.raises(ValueError, match="exactly match registration"):
        admission.validate_config(config)
