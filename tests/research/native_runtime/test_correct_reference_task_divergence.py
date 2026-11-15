"""Offline object-state matching and common-prefix comparison must be unambiguous."""

import numpy as np
import pytest

from scripts.verification.reference_ports.audit_correct_reference_task_divergence import (
    compare_case,
    joint_layout,
    state_position_columns,
)


def fixture():
    rng = np.random.default_rng(42)
    states = rng.normal(size=(5, 12))
    rows, periods = [], []
    for i in range(4):
        rows.append(
            dict(
                step=i,
                objects={"target": dict(position=states[i + 1, 4:7].copy(), grasp_contact=False)},
                gripper=dict(preview=dict(original_reference=0.5)),
                tcp_error=np.zeros(6),
                command=np.zeros(7),
                opening=0.5,
                success=False,
            )
        )
        periods.append(
            dict(
                step=i,
                actual_state=states[i + 1].copy(),
                tcp_error=np.zeros(6),
                command=np.zeros(7),
                completed_feedback=dict(score=dict(opening_reference=0.5, opening=0.5)),
                target_grasp=False,
                task_success=False,
            )
        )
    return rows, states, periods


def test_unique_object_state_mapping_rejects_ambiguous_columns():
    rows, states, _ = fixture()
    positions = np.asarray([row["objects"]["target"]["position"] for row in rows])
    assert state_position_columns(states[1:], positions)["columns"] == [4, 5, 6]
    states[1:, 8:11] = states[1:, 4:7]
    with pytest.raises(ValueError, match="exactly one"):
        state_position_columns(states[1:], positions)


def test_model_layout_tracks_joint_order_and_body_membership():
    first = '<mujoco><worldbody><body name="a"><joint name="j1"/><joint name="j2"/></body></worldbody></mujoco>'
    reordered = '<mujoco><worldbody><body name="a"><joint name="j2"/><joint name="j1"/></body></worldbody></mujoco>'
    moved = '<mujoco><worldbody><body name="a"><joint name="j1"/></body><body name="b"><joint name="j2"/></body></worldbody></mujoco>'
    assert joint_layout(first) != joint_layout(reordered)
    assert joint_layout(first) != joint_layout(moved)


def test_common_prefix_detects_object_divergence_with_identical_gripper_commands():
    rows, states, periods = fixture()
    periods[2]["actual_state"][4] += 0.01
    result = compare_case(rows, states, periods, "target")
    assert result["object_position_difference_m"]["first_above_1mm"] == 2
    assert result["object_position_difference_m"]["maximum"] == pytest.approx(0.01)
    assert result["gripper_command_different_periods"] == 0
    periods[0]["completed_feedback"]["score"]["opening_reference"] = 0.6
    with pytest.raises(ValueError, match="reference"):
        compare_case(rows, states, periods, "target")
