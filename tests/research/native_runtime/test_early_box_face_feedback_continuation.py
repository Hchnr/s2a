"""Both branches share the same pre-dispatch limits and checker isolation."""

import subprocess
import sys

import numpy as np

from scripts.simulation.runtime.early_box_face_feedback_continuation import common_stop


def test_common_stop_checks_arm_and_finger_before_either_policy():
    q = np.zeros(7)
    limits = np.tile([-1.0, 1.0], (7, 1))
    assert common_stop(q, limits, False, -0.03, [-0.04, 0], joint_buffer=0.05) is None
    assert (
        common_stop(q, limits, True, -0.03, [-0.04, 0], joint_buffer=0.05)["stop"]["reason"]
        == "observed_joint_buffer_loss"
    )
    assert (
        common_stop(q, limits, False, -0.041, [-0.04, 0], joint_buffer=0.05)["stop"]["reason"]
        == "finger_joint_outside_declared_range"
    )
    q[0] = 0.96
    assert (
        common_stop(q, limits, False, -0.03, [-0.04, 0], joint_buffer=0.05)["stop"]["reason"]
        == "observed_joint_buffer_loss"
    )


def test_early_checker_import_keeps_both_policies_unloaded():
    code = """\
from scripts.simulation.runtime.early_box_face_feedback_continuation import checker_isolation
checker_isolation()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
