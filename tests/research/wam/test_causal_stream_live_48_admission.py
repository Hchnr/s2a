"""The long live run is admitted only from its exact prior evidence."""

import copy

import pytest

from scripts.wam.tracking.preview import causal_stream_live_48_admission as admission


def test_prior_live_audit_and_closed_physics_are_bound():
    inputs = admission.load_inputs()
    assert inputs["live_config"]["max_windows"] == inputs["live_config"]["max_model_requests"] == 6
    assert inputs["live_config"]["max_native_steps_per_phase"] == 3200
    assert [record["step"] for record in inputs["records"]] == [80, 88]
    assert (
        admission.input_receipt(inputs)["long_refs"]["prior_live_audit"]["sha256"]
        == admission.FIXED["prior_live_audit_sha256"]
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_windows", 7),
        ("max_model_requests", 7),
        ("max_denoise_calls", 211),
        ("automatic_retry", True),
        ("seed", 43),
    ],
)
def test_any_changed_budget_or_seed_is_rejected(key, value):
    config = copy.deepcopy(admission.FIXED)
    config[key] = value
    with pytest.raises(ValueError, match="budgets"):
        admission.validate_config(config)
