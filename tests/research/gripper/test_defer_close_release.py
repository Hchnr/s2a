"""The release proposal must be causal, bounded and independently reconstructable."""

import copy

import numpy as np
import pytest

from s2a.research.gripper.defer_close_release import propose_deferred_close
from scripts.verification.reference_ports.check_defer_close_release import independent_proposal, verify_proposal


def sample():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    poses[:, 0, 3] = np.arange(40) * 0.003
    openings = np.r_[np.linspace(0.1, 0.4, 7), np.full(33, 0.4)]
    return poses, openings, 0, np.eye(4), 0.2, np.array([-0.7, 0.7]), 1.0, True


def test_defers_one_closing_command_only_with_all_fixed_guards():
    args = sample()
    candidate = propose_deferred_close(*args)
    checked = independent_proposal(*args)
    assert candidate["deferred"] and candidate["command"] == 0
    assert verify_proposal(candidate, checked) < 1e-10
    changed = copy.deepcopy(candidate)
    changed["command"] = 1.0
    with pytest.raises(ValueError, match="decision differs"):
        verify_proposal(changed, checked)


@pytest.mark.parametrize("changed", ["motor", "command", "latch", "pose", "opening", "short"])
def test_any_missing_condition_keeps_original_command(changed):
    poses, openings, index, measured, opening, motor, baseline, closing = sample()
    if changed == "motor":
        motor = np.array([0.0, 0.0])
    elif changed == "command":
        baseline = -1.0
    elif changed == "latch":
        closing = False
    elif changed == "pose":
        poses[:, 0, 3] = np.arange(40) * 0.001
    elif changed == "opening":
        openings[3] = openings[2] - 0.02
    else:
        poses, openings = poses[:6], openings[:6]
    args = (poses, openings, index, measured, opening, motor, baseline, closing)
    candidate, checked = propose_deferred_close(*args), independent_proposal(*args)
    assert not candidate["deferred"] and candidate["command"] == baseline
    assert verify_proposal(candidate, checked) < 1e-10


def test_future_beyond_generated_six_step_window_cannot_change_decision():
    args = sample()
    original = propose_deferred_close(*args)
    poses, openings = args[0].copy(), args[1].copy()
    poses[7:, 0, 3] += 10
    openings[7:] = 0.0
    changed = propose_deferred_close(poses, openings, *args[2:])
    assert changed == original


def test_rejects_ambiguous_motor_and_closing_state():
    args = list(sample())
    args[5] = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="antisymmetric"):
        propose_deferred_close(*args)
    args = list(sample())
    args[-1] = 1
    with pytest.raises(ValueError, match="original decision"):
        propose_deferred_close(*args)
