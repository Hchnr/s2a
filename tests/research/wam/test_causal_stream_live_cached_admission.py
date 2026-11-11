"""The cached live trial must bind its independent physical evidence and budget."""

import copy

import pytest

from scripts.wam.tracking.preview import causal_stream_live_cached_admission as admission


def test_cached_live_inputs_bind_the_prior_audit_and_original_model_budget():
    inputs = admission.load_inputs()
    assert inputs["cached_config"]["max_model_requests"] == inputs["live_config"]["max_model_requests"] == 6
    assert inputs["cached_config"]["max_denoise_calls"] == 210
    assert inputs["cached_audit_ref"]["sha256"] == admission.FIXED["prior_cache_audit_sha256"]
    assert inputs["long_config_ref"] == inputs["cached_config_ref"]


@pytest.mark.parametrize(("key", "value"), [("max_model_requests", 7), ("max_denoise_calls", 211), ("seed", 43)])
def test_cached_trial_cannot_expand_its_budget(key, value):
    config = copy.deepcopy(admission.FIXED)
    config[key] = value
    with pytest.raises(ValueError, match="budgets"):
        admission.validate_cached_config(config)
