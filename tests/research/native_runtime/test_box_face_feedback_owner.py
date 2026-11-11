"""The owner enforces physical budgets and reports matched contact samples."""

import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime.run_box_face_feedback import continuation_budget, summarize_periods


def test_runtime_limit_includes_replayed_prefix_and_six_new_periods():
    config = {"max_control_steps_per_phase": 126, "max_native_steps_per_phase": 3150}
    assert continuation_budget(80, config) == 46
    with pytest.raises(ValueError):
        continuation_budget(79, config)
    with pytest.raises(ValueError):
        continuation_budget(80, {**config, "max_native_steps_per_phase": 3200})


def test_contact_counts_use_active_inner_samples_and_original_targets():
    period = dict(
        tcp_error=[0.02, 0, 0, 0, 0, 0],
        pose_caps_satisfied=False,
        completed_feedback={"projected_pose_caps_satisfied": True},
        task_success=True,
        target_grasp=False,
        inner_below_buffer_samples=0,
    )
    conditions = dict(
        finger_joint_range_met=True,
        position_bound_met=True,
        rotation_bound_met=True,
        slide_vertex_residual_m=1e-12,
        slide_axis_change_tcp=1e-13,
    )
    empty = dict(
        contacts=[],
        controller_raw_torque_nm=np.zeros(7),
        commanded_torque_nm=np.zeros(7),
        bounded_sample_conditions=conditions,
    )
    samples = [dict(empty) for _ in range(25)]
    samples[0] = {
        **empty,
        "contacts": [
            dict(
                geom1="table_collision",
                geom2="gripper0_finger2_collision",
                active_constraint=True,
                kind="robot_environment",
            )
        ],
    }
    samples[1] = {
        **empty,
        "contacts": [
            dict(
                geom1="flat_stove_1_collision_burner",
                geom2="gripper0_finger2_collision",
                active_constraint=False,
                kind="robot_environment",
            )
        ],
    }
    result = summarize_periods([period], samples, has_slide_conditions=True)
    assert result["original_pose_caps_satisfied"] == 0 and result["selected_pose_caps_satisfied"] == 1
    assert (
        result["table_finger_active_contact_inner_samples"]
        == result["robot_environment_active_contact_inner_samples"]
        == 1
    )
    assert result["stove_finger_active_contact_inner_samples"] == 0
    assert result["mean_original_position_error_m"] == pytest.approx(0.02)
    with pytest.raises(ValueError):
        summarize_periods([period], samples[:-1], has_slide_conditions=True)


def test_checker_owner_import_does_not_load_candidate_policy():
    code = """\
from scripts.simulation.runtime.run_box_face_feedback import checker_isolation
checker_isolation()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
