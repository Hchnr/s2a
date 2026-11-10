"""Check the causal inputs, thresholds, and incomplete endpoint comparisons."""

import copy

import numpy as np
import pytest

from s2a.research.joint_control.limits.quality_warning import combined_warning, summarize_observations


def observation(endpoint, error, *, velocity=False, margin=0.2):
    return {
        "endpoint": endpoint,
        "previous_tcp_error": error,
        "previous_pose_caps_satisfied": (
            None if error is None else bool(np.linalg.norm(error[:3]) <= 0.01 and np.linalg.norm(error[3:]) <= 0.05)
        ),
        "minimum_margin_rad": margin,
        "warnings": {"velocity_0p2s": {"alarm": velocity}},
    }


def test_initial_no_history_uses_only_current_velocity_flag():
    assert combined_warning(False, None)["alarm"] is False
    assert combined_warning(True, None)["reasons"] == ["velocity"]
    assert combined_warning(True, None)["previous_position_error_m"] is None


@pytest.mark.parametrize(
    "error,expected",
    [
        ([0.003, 0.004, 0, 0, 0, 0], ["position"]),
        ([0, 0, 0, 0.015, 0.020, 0], ["rotation"]),
        ([0.005, 0, 0, 0.025, 0, 0], ["position", "rotation"]),
    ],
)
def test_half_cap_warning_uses_vector_norm_and_includes_boundary(error, expected):
    assert combined_warning(False, error)["reasons"] == expected


def test_alarm_ignores_unavailable_future_and_unrelated_fields():
    old = [observation(0, None), observation(1, [0.006, 0, 0, 0, 0, 0])]
    changed = copy.deepcopy(old)
    changed[1]["future_state"] = {"privileged_object_pose": [9, 9, 9]}
    changed.append(observation(2, [0.5, 0, 0, 0, 0, 0], margin=-1))
    assert summarize_observations(old)["observations"] == summarize_observations(changed)["observations"][:2]


def test_same_endpoint_is_not_a_lead_and_boundary_is_not_failure():
    rows = [
        observation(0, None),
        observation(1, [0.01, 0, 0, 0, 0, 0], margin=0.05),
        observation(2, [0.0101, 0, 0, 0, 0, 0], margin=0.049),
    ]
    result = summarize_observations(rows)
    assert result["first_quality_failure_endpoint"] == result["first_joint_buffer_loss_endpoint"] == 2
    assert result["rules"]["combined"]["relative_to_first_quality_failure"] == "before"
    rows[1]["warnings"]["velocity_0p2s"]["alarm"] = False
    rows[2]["warnings"]["velocity_0p2s"]["alarm"] = True
    assert summarize_observations(rows)["rules"]["velocity"]["relative_to_first_quality_failure"] == "same_endpoint"


def test_missing_terminal_observation_and_no_warning_are_not_success():
    result = summarize_observations([observation(0, None), observation(1, [0] * 6)])
    assert result["first_quality_failure_endpoint"] is None
    assert result["unobserved_endpoint_after_prefix"] == 2
    assert result["rules"]["combined"]["first_warning_endpoint"] is None
    assert result["rules"]["combined"]["relative_to_first_joint_buffer_loss"] == "event_not_observed"


@pytest.mark.parametrize("error", [[0] * 5, [float("nan")] * 6])
def test_invalid_or_nonfinite_previous_error_is_rejected(error):
    with pytest.raises(ValueError):
        combined_warning(False, error)


def test_missing_past_non_boolean_velocity_and_noncontiguous_inputs_are_rejected():
    with pytest.raises(ValueError, match="boolean"):
        combined_warning(1, [0] * 6)
    with pytest.raises(ValueError, match="Only endpoint zero"):
        summarize_observations([observation(0, None), observation(1, None)])
    with pytest.raises(ValueError, match="contiguous"):
        summarize_observations([observation(0, None), observation(2, [0] * 6)])


def test_inconsistent_logged_pose_caps_are_rejected():
    rows = [observation(0, None), observation(1, [0.02, 0, 0, 0, 0, 0])]
    rows[1]["previous_pose_caps_satisfied"] = True
    with pytest.raises(ValueError, match="disagree"):
        summarize_observations(rows)
