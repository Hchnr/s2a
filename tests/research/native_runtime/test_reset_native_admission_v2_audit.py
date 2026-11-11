"""Independent reset audit rejects action dispatch and stale observations."""

import copy

import pytest

from scripts.verification.response_models.audit_reset_native_admission_v2 import validate_worker


def worker():
    zero, three = {"attempted": 0, "completed": 0}, {"attempted": 3, "completed": 3}
    binding = {"sim_time": 0.25, "sim_state_sha256": "initial"}
    return {
        "complete": True,
        "passed": True,
        "control_qualified": False,
        "case_index": 0,
        "case": {"case_index": 0},
        "runtime": {
            "counters": {
                "model_construct": three,
                "context": copy.deepcopy(three),
                "mj_step": zero,
                "mj_step2": copy.deepcopy(zero),
            },
            "reset_endpoint_comparison": {
                "state_bytes_equal": True,
                "max_field_difference": 0,
                "recorded_state_sha256": "initial",
                "actual_state_sha256": "initial",
            },
            "reset_render": {
                "current_endpoint": {**binding, "endpoint_step": 0},
                "render_before": binding,
                "render_after": dict(binding),
            },
            "reset_observation_verified": True,
            "reset_image_match": True,
            "cleanup_complete": True,
            "behavioural_replay_verified": False,
            "previous_saved_command": None,
            "prefix_control_steps": 0,
            "control_steps_attempted": 0,
            "control_steps_completed": 0,
            "checked_endpoints": 1,
            "exact_state_endpoints": 1,
            "canvas_thread_scope": {"before": 1, "requested": 2, "inside": 2, "after": 1},
        },
    }


def test_audit_requires_zero_actions_and_exact_render_binding():
    value = worker()
    assert validate_worker(value, 0)["current_endpoint"]["endpoint_step"] == 0
    for field, bad in (
        ("control_steps_attempted", 1),
        ("reset_image_match", False),
        ("previous_saved_command", [0] * 7),
        ("behavioural_replay_verified", True),
    ):
        changed = copy.deepcopy(value)
        changed["runtime"][field] = bad
        with pytest.raises(ValueError, match="zero-step"):
            validate_worker(changed, 0)
    changed = copy.deepcopy(value)
    changed["runtime"]["reset_render"]["render_after"]["sim_state_sha256"] = "other"
    with pytest.raises(ValueError, match="state binding"):
        validate_worker(changed, 0)
