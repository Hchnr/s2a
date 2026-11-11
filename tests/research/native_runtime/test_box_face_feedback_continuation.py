"""Replay rejects changed causal observations without recomputing a policy."""

import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime.box_face_feedback_continuation import replay_decision


def test_checker_replays_only_matching_measured_decisions():
    current = dict(
        vertices_tcp=np.array([[0, 0, 0]]), tcp=np.eye(4), slide_axis_tcp=np.array([1.0, 0, 0]), joint_position_m=-0.03
    )
    saved = dict(
        command=[0, 0, 0, 0, 0, 0, -0.6],
        decision=dict(
            decision_vertices_tcp=[[0, 0, 0]],
            decision_tcp=np.eye(4).tolist(),
            decision_slide_axis_tcp=[1, 0, 0],
            decision_joint_position_m=-0.03,
            selected_target_tcp=np.eye(4).tolist(),
        ),
    )
    result = replay_decision(current, saved)
    assert result["newly_selected"] is False
    np.testing.assert_array_equal(result["command"], saved["command"])
    current["joint_position_m"] = -0.02
    with pytest.raises(AssertionError):
        replay_decision(current, saved)


def test_checker_continuation_does_not_import_candidate_policy():
    code = """\
from scripts.simulation.runtime.box_face_feedback_continuation import checker_isolation
checker_isolation()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
