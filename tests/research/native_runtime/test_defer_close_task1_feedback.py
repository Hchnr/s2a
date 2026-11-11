"""Independent replay checks the dispatched gripper decision and its inputs."""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import run_defer_close_task1_feedback as feedback
from tests.research.gripper.test_defer_close_release import sample


def test_checker_recomputes_recorded_predicate_from_current_state(monkeypatch):
    from s2a.research.gripper.defer_close_release import propose_deferred_close

    poses, openings, index, tcp, measured, motor, command, closing = sample()
    proposal = propose_deferred_close(poses, openings, index, tcp, measured, motor, command, closing)
    case = {"controller": {"source_from_site": np.eye(4)}, "tool": np.eye(4)}
    selected = {"targets": poses, "openings": openings, "cursor": 0}
    period = {
        "command": np.zeros(7),
        "decision": {
            "gripper_decision": {
                "baseline": {"command": command, "latch": {"closing_episode": closing}},
                "command": 0.0,
                "deferral": proposal,
            }
        },
    }
    monkeypatch.setattr(feedback, "checker_isolation", lambda: None)
    monkeypatch.setattr(feedback, "current_joint_site_state", lambda _env: (None, None, np.zeros((6, 7))))
    monkeypatch.setattr(feedback, "current_robot_dynamics", lambda _env, _jac: (None, None, tcp))
    monkeypatch.setattr(feedback.base, "measured_gripper", lambda _env: {"opening": measured, "motor_target": motor})
    assert feedback.check_recorded_proposal(None, case, selected, 0, period) < 1e-10
    altered = copy.deepcopy(period)
    altered["command"][6] = 1.0
    with pytest.raises(ValueError, match="dispatched command"):
        feedback.check_recorded_proposal(None, case, selected, 0, altered)


def test_clean_checker_entry_does_not_import_candidate_policies():
    source = (
        "import sys; import scripts.simulation.runtime.run_defer_close_task1_feedback as m; "
        "m.checker_isolation(); assert all(name not in sys.modules for name in m.FORBIDDEN)"
    )
    completed = subprocess.run([sys.executable, "-c", source], env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
