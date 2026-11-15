"""Refresh reconstruction must use the preceding completed witness."""

import pytest

from scripts.tracking.response_models.audit_refresh_margin_events import check_signal


def event(position, rotation):
    errors = [[0.0] * 6 for _ in range(8)]
    errors[1] = [position, 0.0, 0.0, rotation, 0.0, 0.0]
    period = {
        "step": 4,
        "endpoint_receipt": {"endpoint_step": 5},
        "actual_inner": [{}] * 25,
        "completed_feedback": {
            "policy_history": {"past_inner_error": {"last_completed_step": 4, "future_guarantee": False}}
        },
        "decision": {
            "quality": {
                "verification": {
                    "passed": True,
                    "first_pose_caps_satisfied": True,
                    "hard_pose_prefix": 2,
                    "prefix_pose_caps_satisfied": True,
                    "prefix_pose_caps_enforced": True,
                    "tcp_errors": errors,
                }
            }
        },
    }
    signal = {
        "current_step": 5,
        "replan": position >= 0.0095 - 1e-10 or rotation >= 0.047 - 1e-10,
        "predicted_next_position_error_m": position,
        "predicted_next_rotation_error_rad": rotation,
        "position_margin_limit_m": 0.0095,
        "rotation_margin_limit_rad": 0.047,
    }
    signal["reason"] = "quality_margin_consumed" if signal["replan"] else "quality_margin_available"
    return period, signal


def test_signal_uses_next_pose_of_just_completed_action():
    period, signal = event(0.0095, 0.02)
    result = check_signal(period, signal)
    assert result == {"reason": "quality_margin_consumed", "near_position": True, "near_rotation": False}
    period["endpoint_receipt"]["endpoint_step"] = 6
    with pytest.raises(ValueError, match="fully completed"):
        check_signal(period, signal)


def test_signal_rejects_unverified_or_tampered_predictions():
    period, signal = event(0.002, 0.01)
    assert check_signal(period, signal)["reason"] == "quality_margin_available"
    signal["predicted_next_position_error_m"] = 0.003
    with pytest.raises(ValueError, match="magnitudes"):
        check_signal(period, signal)
    signal["predicted_next_position_error_m"] = 0.002
    period["completed_feedback"]["policy_history"]["past_inner_error"]["future_guarantee"] = True
    with pytest.raises(ValueError, match="causal"):
        check_signal(period, signal)


def test_first_only_audit_requires_refresh():
    period, signal = event(0.002, 0.01)
    period["decision"]["quality"]["verification"]["hard_pose_prefix"] = 1
    signal = {"current_step": 5, "replan": True, "reason": "no_audited_second_pose"}
    assert check_signal(period, signal)["reason"] == "no_audited_second_pose"
