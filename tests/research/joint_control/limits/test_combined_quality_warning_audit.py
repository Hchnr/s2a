"""Guard against accidentally changing the reviewed case identity or prefix."""

import pytest

from scripts.tracking.joint_control.limits.audit_combined_quality_warning import checked_case


def case():
    return {
        "case": {"case_index": 2},
        "original_endpoint_outside_count": 0,
        "rules": {"velocity_0p2s": {"first_alarm_endpoint": None}},
        "observations": [
            {
                "endpoint": 0,
                "previous_tcp_error": None,
                "previous_pose_caps_satisfied": None,
                "minimum_margin_rad": 0.5,
                "warnings": {"velocity_0p2s": {"alarm": False}},
            },
            {
                "endpoint": 1,
                "previous_tcp_error": [0.006, 0, 0, 0, 0, 0],
                "previous_pose_caps_satisfied": True,
                "minimum_margin_rad": 0.4,
                "warnings": {"velocity_0p2s": {"alarm": False}},
            },
        ],
    }


def test_reviewed_prefix_and_original_velocity_summary_are_reused():
    result = checked_case(case(), 2, expected_endpoints=2)
    assert result["rules"]["combined"]["first_warning_endpoint"] == 1
    assert result["rules"]["velocity"]["first_warning_endpoint"] is None


def test_wrong_case_or_incomplete_prefix_is_rejected():
    with pytest.raises(ValueError, match="case order"):
        checked_case(case(), 1, expected_endpoints=2)
    with pytest.raises(ValueError, match="Complete original"):
        checked_case(case(), 2, expected_endpoints=3)


def test_changed_original_velocity_summary_is_rejected():
    item = case()
    item["rules"]["velocity_0p2s"]["first_alarm_endpoint"] = 1
    with pytest.raises(ValueError, match="velocity warning identity"):
        checked_case(item, 2, expected_endpoints=2)
