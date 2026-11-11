"""The separate replay entrypoint never loads an action-selection policy."""

import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import replay_cached_calibration_gripper as replay


def test_clean_entrypoint_does_not_import_forbidden_policies():
    command = [
        sys.executable,
        "-c",
        "from scripts.simulation.runtime import replay_cached_calibration_gripper as replay; replay.require_independence()",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_forbidden_import_is_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, replay.FORBIDDEN[-1], object())
    with pytest.raises(ValueError, match="checker imported candidate"):
        replay.require_independence()


def test_independent_reference_and_post_step_scores():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    chunk = dict(anchor_tcp=np.eye(4), anchor_opening=0.8, effective_targets_tcp=poses, openings=np.full(40, 0.5))
    candidate = dict(pose_coefficients=np.zeros((7, 6)), opening_coefficients=np.array([-0.1, 0, 0, 0, 0, 0, 0]))
    config = dict(
        pi_margin_rad=0.0001,
        position_scale_m=0.1,
        rotation_scale_rad=0.5,
        position_cap_m=0.05,
        rotation_cap_rad=0.25,
        opening_cap=0.25,
    )
    refs = replay.independent_references(chunk, candidate, config)
    actual = np.eye(4)
    actual[0, 3] = 0.01
    scores = replay.independent_scores(refs, actual, 0.45, 0)
    assert scores["raw"]["tcp_error"][0] == pytest.approx(-0.01)
    assert scores["raw"]["opening_error"] == pytest.approx(-0.05)
    assert scores["calibrated"]["opening_error"] == pytest.approx(0.05)


def test_checker_replays_only_the_recorded_command():
    script = (
        "import numpy as np; "
        "from scripts.simulation.runtime import replay_cached_calibration_gripper as r; "
        "expected = [{'command': [0.1] * 7}]; "
        "decision = r.common.checker_action(expected)(None, None, None, None, 0); "
        "assert not decision['newly_selected']; "
        "np.testing.assert_array_equal(decision['command'], expected[0]['command'])"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
