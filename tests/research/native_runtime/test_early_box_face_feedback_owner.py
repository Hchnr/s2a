"""The earlier owner compares identical initial controller state and targets."""

import copy
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime.run_early_box_face_feedback import compare_common_reference


def branch():
    start = dict(
        state_sha256="state",
        time_s=5.9,
        current_tcp=np.eye(4).tolist(),
        current_vertices_tcp=[[0, 0, 0]],
        previous_command=[0] * 7,
        controller_goal_position=[0] * 3,
        controller_goal_rotation=np.eye(3).tolist(),
        gripper_motor_target=[0, 0],
    )
    return dict(
        branch_start=start,
        reference_step=112,
        reference_image_ref={"sha256": "image"},
        reference_response_ref={"sha256": "response"},
        original_targets=[np.eye(4).tolist()] * 8,
    )


def test_common_reference_rejects_changed_targets_or_start():
    original = branch()
    compare_common_reference(original, copy.deepcopy(original))
    changed = copy.deepcopy(original)
    changed["reference_image_ref"]["sha256"] = "other"
    with pytest.raises(ValueError):
        compare_common_reference(original, changed)
    changed = copy.deepcopy(original)
    changed["branch_start"]["previous_command"][0] = 0.1
    with pytest.raises(AssertionError):
        compare_common_reference(original, changed)
    changed = copy.deepcopy(original)
    changed["original_targets"][0][0][0] = 2
    with pytest.raises(AssertionError):
        compare_common_reference(original, changed)


def test_owner_import_keeps_checker_policy_isolated():
    code = """\
from scripts.simulation.runtime.run_early_box_face_feedback import checker_isolation
checker_isolation()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
