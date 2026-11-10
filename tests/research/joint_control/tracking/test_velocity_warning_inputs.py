"""Earlier warnings and predecision inputs remain causal and bounded."""

import copy

import numpy as np
import pytest

from s2a.research.joint_control.tracking.velocity_warning_inputs import (
    current_recorded_decision,
    first_velocity_warning,
)


def xml():
    return (
        "<mujoco><worldbody><body>"
        + "".join(f'<joint name="robot0_joint{i}" type="hinge" range="-1 1"/>' for i in range(1, 8))
        + "</body></worldbody></mujoco>"
    )


def test_first_warning_uses_no_future_endpoints_and_keeps_actual_buffer():
    states = np.zeros((3, 15))
    states[-1, 1] = 0.7
    states[-1, 8] = 1.4
    result = first_velocity_warning(states, xml())
    assert result["endpoint"] == 2 and result["warning"]["alarm"] is True
    assert result["minimum_margin_rad"] == pytest.approx(0.3)
    states[1] = states[-1]
    with pytest.raises(ValueError, match="first causal"):
        first_velocity_warning(states, xml())


def test_recorded_decision_ignores_future_observation_and_checks_executed_plan():
    plan = np.tile([0.1, 0, 0, 0.001, 0, 0], (8, 1))
    row = {
        "step": 5,
        "chunk_index": 0,
        "chunk_physical_step": 5,
        "command": np.r_[plan[0], 0],
        "tracking_refinement": {"planned_commands": plan, "nominal_response": np.zeros(12)},
        "coupled_dynamics": {
            "mass": np.eye(7),
            "jacobian": np.zeros((6, 7)),
            "robot_joint_position": np.zeros(7),
            "robot_joint_velocity": np.zeros(7),
        },
        "qualified_forward_input": {"x_site_pose": [np.eye(4)], "x_site_twist": [np.zeros(6)]},
        "after_source": [100],
        "objects": {"privileged_future": True},
    }
    selected = current_recorded_decision(row, step=5, previous_command=np.zeros(7))
    changed = copy.deepcopy(row)
    changed["after_source"], changed["objects"] = [999], {"privileged_future": False}
    again = current_recorded_decision(changed, step=5, previous_command=np.zeros(7))
    for key in ("plan", "nominal_response", "mass", "jacobian", "q", "velocity", "site", "twist"):
        np.testing.assert_array_equal(selected[key], again[key])
    changed["command"][0] = 0.2
    with pytest.raises(ValueError, match="original command"):
        current_recorded_decision(changed, step=5, previous_command=np.zeros(7))
