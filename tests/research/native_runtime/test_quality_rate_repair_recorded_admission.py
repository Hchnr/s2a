"""Recorded references and completed physics are bound before new feedback."""

import copy

import pytest

from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as admission


def test_six_real_saved_requests_and_45_replayed_periods_are_bound():
    inputs = admission.load_inputs()
    assert [row["step"] for row in inputs["records"]] == [80, 88, 96, 104, 112, 120]
    assert len(inputs["prior_periods"]) == 45
    assert all(row["historical_denoise_calls"] == 35 for row in inputs["records"])
    assert len({row["response_sha256"] for row in inputs["records"]}) == 6


def test_completed_physics_guard_does_not_read_task_labels():
    row = admission.load_inputs()["prior_periods"][-1]
    changed = copy.deepcopy(row)
    changed["task_success"] = not row["task_success"]
    changed["target_grasp"] = not row["target_grasp"]
    assert admission.matched_prior_period(changed, row)
    changed["command"][0] += 0.01
    with pytest.raises(AssertionError):
        admission.matched_prior_period(changed, row)


@pytest.mark.parametrize(
    ("key", "value"), [("max_windows", 7), ("new_model_requests", 1), ("max_native_steps_per_phase", 3201)]
)
def test_unregistered_physics_or_model_work_is_rejected(key, value):
    config = copy.deepcopy(admission.FIXED)
    config[key] = value
    with pytest.raises(ValueError, match="physical budget"):
        admission.validate_config(config)
