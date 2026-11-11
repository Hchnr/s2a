"""Replay only actually completed stream actions in a separate synthetic environment."""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from s2a.research.native_runtime.streaming_replay_48 import replay_streaming_continuation
from tests.research.native_runtime.test_streaming_continuation import execute, fake_case


def cases(tmp_path, *, windows=3, producer_options=None, runtime_options=None):
    producer_dir, checker_dir = tmp_path / "producer", tmp_path / "checker"
    producer_dir.mkdir()
    checker_dir.mkdir()
    producer = fake_case(producer_dir, windows=windows, **(runtime_options or {}))
    checker = fake_case(checker_dir, windows=windows)
    stream = execute(producer, **(producer_options or {}))
    return producer, checker, stream


def replay(checker, stream, *, targets=None, source_observations=None):
    if targets is None:
        targets = [np.tile(np.eye(4), (8, 1, 1)) for _ in stream["segments"]]
    return replay_streaming_continuation(
        checker.env,
        checker.terminal,
        checker.observers,
        checker.receipt,
        producer_stream=stream,
        target_chunks=targets,
        source_observations=source_observations,
        max_windows=checker.windows,
        previous_command=np.zeros(7),
        tool=np.eye(4),
        target_object="target",
        state_reader=checker.state_reader,
        task_observer=checker.task_observer,
    )


def test_independent_checker_replays_three_complete_mock_windows(tmp_path):
    producer, checker, stream = cases(tmp_path)
    result = replay(checker, stream)
    assert result["complete"] and result["reported_producer_stop"] is None
    assert result["stop_reason_independently_verified"] is False
    assert result["control_steps_attempted"] == result["control_steps_completed"] == 24
    assert producer.sim.inner == checker.sim.inner == 600
    assert checker.receipt["newly_selected_commands"] == 0
    assert [row["start_step"] for row in result["segments"]] == [80, 88, 96]


def test_checker_replays_only_completed_actions_and_does_not_certify_policy_stop(tmp_path):
    _producer, checker, stream = cases(tmp_path, producer_options={"stop_at": 82})
    result = replay(checker, stream)
    assert result["complete"] and result["control_steps_completed"] == 2
    assert result["reported_producer_stop"]["reason"] == "quality_no_witness"
    assert result["stop_reason_independently_verified"] is False
    assert checker.sim.inner == 50


def test_checker_accepts_explicit_stop_before_first_request_without_dispatch(tmp_path):
    checker = fake_case(tmp_path)
    stream = {
        "segments": [],
        "complete": False,
        "stop": {"reason": "observed_prefix_joint_buffer_loss", "at_step": 80, "next_action_dispatched": False},
        "control_steps_attempted": 0,
        "control_steps_completed": 0,
        "feedback_periods_committed": 0,
    }
    result = replay(checker, stream)
    assert result["complete"] and result["segments"] == []
    assert result["control_steps_completed"] == result["control_steps_attempted"] == checker.sim.inner == 0
    assert result["reported_producer_stop"] == stream["stop"]
    assert result["stop_reason_independently_verified"] is False


def test_checker_rejects_an_empty_stream_with_a_future_stop(tmp_path):
    checker = fake_case(tmp_path)
    stream = {
        "segments": [],
        "complete": False,
        "stop": {"reason": "observed_prefix_joint_buffer_loss", "at_step": 81, "next_action_dispatched": False},
        "control_steps_attempted": 0,
        "control_steps_completed": 0,
        "feedback_periods_committed": 0,
    }
    with pytest.raises(ValueError, match="close every recorded dispatch"):
        replay(checker, stream)
    assert checker.sim.inner == 0


def test_changed_recorded_inner_sample_fails_the_independent_replay(tmp_path):
    _producer, checker, stream = cases(tmp_path)
    tampered = copy.deepcopy(stream)
    tampered["segments"][0]["continuation"]["periods"][2]["actual_inner"][3]["q"][0] = 0.2
    with pytest.raises(AssertionError):
        replay(checker, tampered)
    result = checker.receipt["streaming_replay_partial"]
    assert not result["complete"] and result["control_steps_completed"] == 3
    assert checker.sim.inner == 75


def test_wrong_reference_is_detected_by_recorded_tcp_error(tmp_path):
    _producer, checker, stream = cases(tmp_path)
    targets = [np.tile(np.eye(4), (8, 1, 1)) for _ in stream["segments"]]
    targets[0][0, 0, 3] = 0.2
    with pytest.raises(AssertionError):
        replay(checker, stream, targets=targets)
    assert checker.receipt["streaming_replay_partial"]["control_steps_completed"] == 1


def test_checker_binds_each_decoding_anchor_to_independently_observed_pose(tmp_path):
    _producer, checker, stream = cases(tmp_path)
    observations = [{"source_pose": np.eye(4), "opening": 0.7} for _ in stream["segments"]]
    observations[1]["source_pose"][0, 3] = 0.1
    with pytest.raises(AssertionError):
        replay(checker, stream, source_observations=observations)
    assert checker.sim.inner == 200
    assert checker.receipt["streaming_replay_partial"]["control_steps_completed"] == 8


def test_partial_inflight_dispatch_is_rejected_before_replay(tmp_path):
    producer_dir, checker_dir = tmp_path / "producer", tmp_path / "checker"
    producer_dir.mkdir()
    checker_dir.mkdir()
    producer = fake_case(producer_dir, fail_at=3)
    checker = fake_case(checker_dir)
    with pytest.raises(RuntimeError, match="mid-period"):
        execute(producer)
    with pytest.raises(ValueError, match="in-flight"):
        replay(checker, producer.receipt["streaming_partial"])
    assert checker.sim.inner == 0


def test_incomplete_or_unaccounted_receipt_is_rejected_before_dispatch(tmp_path):
    _producer, checker, stream = cases(tmp_path)
    stream["control_steps_completed"] -= 1
    with pytest.raises(ValueError, match="close every recorded dispatch"):
        replay(checker, stream)
    assert checker.sim.inner == 0


def test_checker_import_does_not_load_candidate_feedback_or_model_adapter():
    code = """
import sys
from s2a.research.native_runtime.streaming_replay import replay_streaming_continuation
for module in (
    's2a.research.native_runtime.streaming_continuation',
    's2a.research.wam.tracking.runtime.causal_stream_reference',
    's2a.research.joint_control.preview.streaming_feedback_boundary',
    's2a.research.joint_control.preview.loaded_gripper_feedback',
):
    assert module not in sys.modules, module
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr
