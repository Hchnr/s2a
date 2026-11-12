"""Validate joint direction and retain attempted LP work on budget stops."""

import numpy as np
import pytest

from scripts.simulation.runtime.analyze_bowl_static_close_feasibility import solve_branch, verify_joint_contract


def test_joint_contract_checks_ranges_and_site_axis_direction():
    xml = """<mujoco><worldbody><body name="gripper0_right_gripper">
      <body name="gripper0_eef"><site name="gripper0_grip_site"/></body>
      <body name="gripper0_leftfinger" quat="0.707107 0 0 0.707107">
        <joint name="gripper0_finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/></body>
      <body name="gripper0_rightfinger" quat="0.707107 0 0 0.707107">
        <joint name="gripper0_finger_joint2" type="slide" axis="0 1 0" range="-0.04 0"/></body>
    </body></worldbody></mujoco>"""
    verify_joint_contract(xml)
    with pytest.raises(AssertionError):
        verify_joint_contract(xml.replace('axis="0 1 0"', 'axis="1 0 0"', 1))
    with pytest.raises(AssertionError):
        verify_joint_contract(xml.replace('range="0 0.04"', 'range="0 0.05"'))


def test_budget_stop_preserves_attempted_work():
    def box(name, x):
        return dict(name=name, center_world_m=[x, 0, 0], rotation_world=np.eye(3), half_size_m=[0.01] * 3)

    boxes = [box("left", -0.05), box("right", 0.05)] + [box(f"target{i}", 0) for i in range(40)]
    point = dict(
        state_sha256="a",
        pad_alignment=dict(box_projections=boxes, site_rotation_world=np.eye(3), jaw_axis_world=[1, 0, 0]),
    )
    period = dict(
        endpoint_receipt=dict(sim_state_sha256="a"),
        tcp_error=np.zeros(6),
        completed_feedback=dict(score=dict(qpos=[0.03, -0.03])),
    )
    config = dict(
        strict_overlap_margin_m=1e-5,
        max_lp_per_branch=1,
        max_refinements_per_branch=1,
        position_gate_m=0.01,
        feasibility_tolerance_m=1e-8,
    )
    progress = {}
    with pytest.raises(ValueError, match="LP budget is exhausted"):
        solve_branch(point, period, config, progress=progress)
    assert progress["lp_attempts"] == 1 and len(progress["records"]) == 1
