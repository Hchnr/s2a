"""Admit only the failed live run and its actually completed actions."""

import copy

import pytest

from scripts.simulation.runtime import causal_endpoint_diagnostic_admission as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def test_failed_live_stream_is_bound_without_rewriting_its_stop():
    inputs = admission.load_inputs()
    assert inputs["branch"]["runtime"]["streaming_partial"]["stop"] is None
    assert inputs["stream"]["diagnostic_derived_stop"] is True
    assert inputs["stream"]["stop"]["at_step"] == 120
    assert len(inputs["chunks"]) == len(inputs["source_observations"]) == 5
    assert inputs["config"]["max_native_steps"] == 3000


@pytest.mark.parametrize("change", ["more_steps", "inflight", "different_failure"])
def test_unaccounted_partial_execution_cannot_become_a_diagnostic_trace(change):
    inputs = admission.load_inputs()
    branch = copy.deepcopy(inputs["branch"])
    if change == "more_steps":
        branch["runtime"]["control_steps_completed"] += 1
    elif change == "inflight":
        branch["runtime"]["streaming_partial"]["segments"][-1]["inflight"] = {"inner_completed": 1}
    else:
        branch["runtime"]["error"]["message"] = "different failure"
    with pytest.raises(ValueError, match="exact failed producer"):
        admission.diagnostic_stream(branch)


def test_missing_actual_denoise_completion_is_rejected():
    inputs = admission.load_inputs()
    model = copy.deepcopy(prefix.read(inputs["refs"]["failed_model"]))
    model["denoise_evidence"][25]["completed"] = False
    with pytest.raises(ValueError, match="35 completed actual denoise"):
        admission.verify_model(inputs["live"], model, inputs["branch"], {})


@pytest.mark.parametrize(("key", "value"), [("max_native_steps", 3025), ("new_model_requests", 1), ("seed", 43)])
def test_diagnostic_cannot_expand_its_budget(key, value):
    config = copy.deepcopy(admission.FIXED)
    config[key] = value
    with pytest.raises(ValueError, match="diagnostic budget"):
        admission.validate_config(config)
