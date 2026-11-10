"""Physical scoring and replay must retain both references and stop evidence."""

import copy
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime import run_causal_event_feedback as runtime


def test_completed_scoring_keeps_both_aligned_references_and_gripper(monkeypatch):
    site = np.eye(4)
    robot = SimpleNamespace(
        _ref_gripper_joint_pos_indexes=[0, 1],
        _ref_gripper_joint_vel_indexes=[0, 1],
        gripper=SimpleNamespace(current_action=np.array([0.5, -0.5])),
    )
    env = SimpleNamespace(
        robots=[robot], sim=SimpleNamespace(data=SimpleNamespace(qpos=np.array([0.04, -0.04]), qvel=np.zeros(2)))
    )
    monkeypatch.setattr(
        runtime, "current_joint_site_state", lambda _env: (np.zeros(7), np.zeros((7, 2)), np.zeros((6, 7)))
    )
    monkeypatch.setattr(runtime, "current_robot_dynamics", lambda _env, _jac: (np.eye(7), np.zeros(7), site))
    old, latest = np.tile(np.eye(4), (40, 1, 1)), np.tile(np.eye(4), (40, 1, 1))
    old[8, 0, 3] = 0.01
    case = {
        "controller": {"source_from_site": np.eye(4)},
        "tool": np.eye(4),
        "references": {
            "remembered": {"poses": old, "openings": np.full(40, 0.2), "cursor": 8},
            "latest": {"poses": latest, "openings": np.full(40, 0.9), "cursor": 0},
        },
    }
    selected = {"spec": {"prefix_steps": 136}}
    score = runtime.score_completed(env, case, selected, "latest", 0, np.zeros(6))
    assert score["selected_absolute_endpoint"] == 137
    assert score["opening_targets"] == {"latest": 0.9, "remembered": 0.2}
    assert np.linalg.norm(score["reference_errors"]["remembered"][:3]) == pytest.approx(0.01)
    assert score["gripper"]["opening"] == pytest.approx(1.0)
    runtime.score_completed(env, case, selected, "latest", 0, np.zeros(6), expected=score)
    tampered = copy.deepcopy(score)
    tampered["gripper"]["motor_target"][0] = 0.0
    with pytest.raises(AssertionError):
        runtime.score_completed(env, case, selected, "latest", 0, np.zeros(6), expected=tampered)


def test_final_prefix_samples_and_buffer_stop_are_replayed_independently():
    samples = [{"time": (i + 1) * 0.002, "q": np.zeros(7), "qvel": np.zeros(7)} for i in range(25)]
    expected = {
        "prefix_step": 135,
        "below_buffer_samples": 1,
        "outside_samples": 0,
        "minimum_margin_rad": 0.04,
        "limits": np.tile([-1.0, 1.0], (7, 1)),
        "samples": samples,
    }
    runtime.check_prefix_inner(expected, copy.deepcopy(expected))
    altered = copy.deepcopy(expected)
    altered["samples"][4]["q"][0] = 0.1
    with pytest.raises(AssertionError):
        runtime.check_prefix_inner(altered, expected)
    continuation = {"periods": [], "stop": None}
    stop = {
        "reason": "observed_joint_buffer_loss",
        "minimum_current_margin_rad": 0.06,
        "at_step": 136,
        "next_action_dispatched": False,
    }
    with pytest.raises(ValueError, match="observable"):
        runtime.reconcile_stop(copy.deepcopy(continuation), stop, start_step=136, current_margin=0.06)
    result = runtime.reconcile_stop(
        copy.deepcopy(continuation), stop, start_step=136, current_margin=0.06, prefix_inner_loss=True
    )
    assert result["buffer_stop_verified_from_replay"] and not result["decision_stop_recorded_not_recomputed"]


def test_quality_stop_is_explicitly_not_resolved_by_action_replay():
    continuation = {"periods": [], "stop": None}
    stop = {"reason": "quality_no_witness", "solver_status": "test", "at_step": 360, "next_action_dispatched": False}
    result = runtime.reconcile_stop(continuation, stop, start_step=360, current_margin=0.8)
    assert result["stop"] == stop and result["decision_stop_recorded_not_recomputed"]


def test_longer_quality_stop_preserves_its_completed_action_count():
    continuation = {"periods": [{} for _ in range(9)], "stop": None}
    stop = {"reason": "quality_no_witness", "solver_status": "test", "at_step": 369, "next_action_dispatched": False}
    result = runtime.reconcile_stop(continuation, stop, start_step=360, current_margin=0.8, horizon=16)
    assert result["stop"] == stop and result["decision_stop_recorded_not_recomputed"]


def test_checker_entry_does_not_import_or_solve_candidate_policies():
    program = (
        "import sys; from scripts.simulation.runtime import run_causal_event_feedback as r; "
        "r.checker_action([]); assert all(name not in sys.modules for name in r.FORBIDDEN_CHECKER_MODULES)"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
