"""Independently replay variable-length segments using synthetic observations."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from s2a.research.native_runtime.adaptive_streaming_replay import replay_adaptive_streaming_continuation
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from tests.research.native_runtime.test_adaptive_stream_reference import fake_service, run_driver, setup_adapter
from tests.research.native_runtime.test_streaming_continuation import fake_case


def cases(tmp_path, *, checker_render_changes=False):
    producer_dir, checker_dir = tmp_path / "producer", tmp_path / "checker"
    producer_dir.mkdir()
    checker_dir.mkdir()
    producer, adapter, channel, model_channel, _events, _owner = setup_adapter(producer_dir)
    errors = []
    thread = threading.Thread(target=fake_service, args=(producer, model_channel, 3, errors))
    try:
        thread.start()
        stream = run_driver(producer, adapter)
    finally:
        channel.close()
        thread.join(timeout=15)
    assert not thread.is_alive() and not errors
    responses = []
    for exchange in producer.receipt["adaptive_model_exchanges"]:
        with open(exchange["response_ref"]["path"]) as handle:
            responses.append(json.load(handle))
    checker = fake_case(checker_dir, windows=3, render_changes=checker_render_changes)
    service = producer.service
    registration = {
        "session": service.session,
        "initial": service.initial,
        "image_root": checker.ledger.image_root,
        "task": service.task,
        "seed": service.seed,
        "model_sha256": service.model_sha256,
        "max_requests": service.max_requests,
        "max_periods": service.max_periods,
        "denoise_steps": service.denoise_steps,
    }
    checker.ledger = AdaptiveStreamLedger(**registration)
    return producer, checker, stream, responses, checker_dir


def replay(checker, stream, responses, checker_dir, *, changed_image=False, change_capture=False):
    def capture(env, observation, index):
        assert set(observation) == {"snapshot", "source_pose", "opening"}
        assert not observation["source_pose"].flags.writeable
        step = observation["snapshot"]["step"]
        data = f"synthetic image at {step}".encode()
        name = f"image{index}.png"
        (checker_dir / name).write_bytes(data)
        if change_capture:
            env.sim.data.time += 0.002
        return {"path": name, "sha256": "f" * 64 if changed_image else hashlib.sha256(data).hexdigest()}

    def compare(actual, expected):
        np.testing.assert_allclose(actual["desired_opening"], expected["desired_opening"], atol=1e-10, rtol=0)

    return replay_adaptive_streaming_continuation(
        checker.env,
        checker.terminal,
        checker.observers,
        checker.receipt,
        ledger=checker.ledger,
        producer_stream=stream,
        responses=responses,
        capture_current=capture,
        previous_command=np.zeros(7),
        tool=np.eye(4),
        target_object="target",
        score_gripper=lambda _env, opening: {"desired_opening": float(opening)},
        compare_score=compare,
        state_reader=checker.state_reader,
        task_observer=checker.task_observer,
    )


def test_checker_replays_two_partial_replans_with_independent_current_images(tmp_path):
    producer, checker, stream, responses, checker_dir = cases(tmp_path)
    result = replay(checker, stream, responses, checker_dir)
    assert result["complete"] and not result["stop_reason_independently_verified"]
    assert result["reported_producer_stop"] == stream["stop"]
    assert (
        result["control_steps_attempted"]
        == result["control_steps_completed"]
        == result["replay_periods_committed"]
        == 12
    )
    assert [row["start_step"] for row in result["segments"]] == [80, 83, 87]
    assert producer.sim.inner == checker.sim.inner == 300
    assert checker.receipt["newly_selected_commands"] == 0
    for original, checked in zip(stream["segments"], result["segments"], strict=True):
        for left, right in zip(original["continuation"]["periods"], checked["continuation"]["periods"], strict=True):
            np.testing.assert_array_equal(left["command"], right["command"])
            np.testing.assert_array_equal(left["actual_state"], right["actual_state"])
            assert right["completed_feedback"]["policy_history"] is None


def test_corrupted_inner_sample_fails_after_the_actual_replayed_action(tmp_path):
    _producer, checker, stream, responses, checker_dir = cases(tmp_path)
    changed = copy.deepcopy(stream)
    changed["segments"][0]["continuation"]["periods"][2]["actual_inner"][3]["q"][0] = 0.2
    with pytest.raises(AssertionError):
        replay(checker, changed, responses, checker_dir)
    partial = checker.receipt["adaptive_replay_partial"]
    assert not partial["complete"] and partial["control_steps_completed"] == 3
    assert partial["replay_periods_committed"] == 2 and checker.ledger.phase == "failed"


def test_stale_second_image_response_stops_before_second_segment_dispatch(tmp_path):
    _producer, checker, stream, responses, checker_dir = cases(tmp_path)
    changed = copy.deepcopy(responses)
    changed[1]["image_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Response image"):
        replay(checker, stream, changed, checker_dir)
    partial = checker.receipt["adaptive_replay_partial"]
    assert partial["control_steps_completed"] == partial["replay_periods_committed"] == 3
    assert checker.sim.inner == 75 and checker.ledger.phase == "failed"


@pytest.mark.parametrize("change", ["inflight", "boundary", "unaccounted", "future_stop", "after_buffer"])
def test_inconsistent_producer_receipt_is_rejected_before_any_replay(tmp_path, change):
    _producer, checker, stream, responses, checker_dir = cases(tmp_path)
    changed = copy.deepcopy(stream)
    if change == "inflight":
        changed["segments"][0]["inflight"] = {"inner_completed": 3}
    elif change == "boundary":
        changed["segments"][0]["continuation"]["stop"]["trigger"]["current_step"] += 1
    elif change == "unaccounted":
        changed["control_steps_completed"] -= 1
    elif change == "future_stop":
        changed["stop"]["at_step"] += 1
    else:
        changed["segments"][0]["continuation"]["periods"][0]["inner_below_buffer_samples"] = 1
    with pytest.raises(ValueError):
        replay(checker, changed, responses, checker_dir)
    assert checker.sim.inner == 0 and checker.ledger.issued_requests == 0


@pytest.mark.parametrize("failure", ["changed_image", "change_capture", "render_changes"])
def test_unbound_capture_or_state_changing_render_never_dispatches(tmp_path, failure):
    _producer, checker, stream, responses, checker_dir = cases(
        tmp_path, checker_render_changes=failure == "render_changes"
    )
    kwargs = {} if failure == "render_changes" else {failure: True}
    with pytest.raises((ValueError, OSError)):
        replay(checker, stream, responses, checker_dir, **kwargs)
    assert checker.sim.inner == 0 and checker.receipt["adaptive_replay_partial"]["control_steps_attempted"] == 0


def test_stop_before_first_action_keeps_one_image_and_no_fabricated_period(tmp_path):
    _producer, checker, stream, responses, checker_dir = cases(tmp_path)
    changed = copy.deepcopy(stream)
    changed["segments"] = changed["segments"][:1]
    row = changed["segments"][0]
    stop = {"reason": "quality_no_witness", "at_step": 80, "next_action_dispatched": False}
    row["continuation"].update(periods=[], steps_completed=0, stop=stop)
    row.pop("boundary")
    changed.update(
        complete=False,
        stop=stop,
        requests_issued=1,
        responses_accepted=1,
        control_steps_attempted=0,
        control_steps_completed=0,
        feedback_periods_committed=0,
    )
    result = replay(checker, changed, responses[:1], checker_dir)
    assert result["complete"] and result["control_steps_completed"] == checker.sim.inner == 0
    assert len(result["segments"]) == 1 and not result["stop_reason_independently_verified"]


def test_final_partial_trigger_can_stop_at_the_request_budget(tmp_path):
    _producer, checker, stream, responses, checker_dir = cases(tmp_path)
    changed = copy.deepcopy(stream)
    changed["segments"] = changed["segments"][:1]
    changed.update(
        complete=False,
        stop={"reason": "request_budget_exhausted", "at_step": 83, "next_action_dispatched": False},
        requests_issued=1,
        responses_accepted=1,
        control_steps_attempted=3,
        control_steps_completed=3,
        feedback_periods_committed=3,
    )
    old = checker.ledger
    checker.ledger = AdaptiveStreamLedger(
        session=old.session,
        initial=old.current,
        image_root=old.image_root,
        task=old.task,
        seed=old.seed,
        model_sha256=old.model_sha256,
        max_requests=1,
        max_periods=old.max_periods,
        denoise_steps=old.denoise_steps,
    )
    result = replay(checker, changed, responses[:1], checker_dir)
    assert result["complete"] and result["control_steps_completed"] == 3
    assert result["reported_producer_stop"]["reason"] == "request_budget_exhausted"
    assert checker.sim.inner == 75 and checker.receipt["newly_selected_commands"] == 0


def test_prefix_stop_without_any_request_replays_zero_actions(tmp_path):
    checker = fake_case(tmp_path)
    old = checker.ledger
    checker.ledger = AdaptiveStreamLedger(
        session=old.session,
        initial=old.current,
        image_root=old.image_root,
        task=old.task,
        seed=old.seed,
        model_sha256=old.model_sha256,
        max_requests=3,
        max_periods=12,
        denoise_steps=old.denoise_steps,
    )
    stream = {
        "segments": [],
        "complete": False,
        "stop": {"reason": "observed_prefix_joint_buffer_loss", "at_step": 80, "next_action_dispatched": False},
        "requests_issued": 0,
        "responses_accepted": 0,
        "control_steps_attempted": 0,
        "control_steps_completed": 0,
        "feedback_periods_committed": 0,
    }
    result = replay(checker, stream, [], tmp_path)
    assert result["complete"] and result["segments"] == []
    assert result["control_steps_completed"] == result["replay_periods_committed"] == checker.sim.inner == 0


def test_checker_import_does_not_load_candidate_policy_or_producer_adapter():
    code = """
import sys
from s2a.research.native_runtime.adaptive_streaming_replay import replay_adaptive_streaming_continuation
for module in (
    's2a.research.native_runtime.adaptive_streaming_continuation',
    's2a.research.wam.tracking.runtime.adaptive_stream_reference',
    's2a.research.joint_control.preview.causal_reference_replan',
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
