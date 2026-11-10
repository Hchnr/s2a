"""Recompute the first warning using only completed and current observations."""

import numpy as np
import pytest

from s2a.research.joint_control.tracking.combined_warning_inputs import first_combined_warning


def xml():
    return (
        "<mujoco><worldbody><body>"
        + "".join(f'<joint name="robot0_joint{i}" type="hinge" range="-1 1"/>' for i in range(1, 8))
        + "</body></worldbody></mujoco>"
    )


def test_completed_error_warns_at_current_endpoint_not_one_period_earlier():
    states, errors = np.zeros((4, 15)), np.zeros((3, 6))
    errors[-1, 0] = 0.006
    result = first_combined_warning(states, errors, xml())
    assert result["endpoint"] == 3
    assert result["warning"]["reasons"] == ["position"]
    assert result["minimum_margin_rad"] == 1.0
    with pytest.raises(ValueError, match="first combined"):
        first_combined_warning(states[:3], errors[:2], xml())


def test_current_velocity_can_trigger_without_quality_error():
    states, errors = np.zeros((3, 15)), np.zeros((2, 6))
    states[-1, 1], states[-1, 8] = 0.8, 1.0
    assert first_combined_warning(states, errors, xml())["warning"]["reasons"] == ["velocity"]


def test_earlier_warning_is_not_reselected_after_seeing_future():
    states, errors = np.zeros((4, 15)), np.zeros((3, 6))
    errors[0, 0], errors[2, 0] = 0.006, 0.02
    with pytest.raises(ValueError, match="first combined"):
        first_combined_warning(states, errors, xml())
    assert first_combined_warning(states[:2], errors[:1], xml())["endpoint"] == 1


def test_warning_already_outside_original_buffer_is_ineligible():
    states, errors = np.zeros((2, 15)), np.zeros((1, 6))
    states[-1, 1], errors[-1, 0] = 0.96, 0.006
    with pytest.raises(ValueError, match="joint buffer"):
        first_combined_warning(states, errors, xml())


def test_incomplete_history_and_invalid_joint_names_are_rejected():
    states, errors = np.zeros((3, 15)), np.zeros((1, 6))
    with pytest.raises(ValueError, match="completed error"):
        first_combined_warning(states, errors, xml())
    errors = np.zeros((2, 6))
    with pytest.raises(ValueError, match="joint addresses"):
        first_combined_warning(states, errors, xml().replace("robot0_joint1", "different"))
