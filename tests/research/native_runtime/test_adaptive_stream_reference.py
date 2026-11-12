"""Test current-image adaptive requests against a synthetic model service."""

import hashlib
import json
import socket
import threading

import numpy as np
import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.repository_paths import repository_root
from s2a.research.joint_control.preview.causal_reference_replan import (
    advance_recorded_segment,
    completed_quality_margin,
)
from s2a.research.native_runtime.adaptive_streaming_continuation import run_adaptive_streaming_continuation
from s2a.research.native_runtime.streaming_continuation import measured_snapshot
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveModelAdmission, AdaptiveStreamLedger
from s2a.research.wam.tracking.runtime.adaptive_stream_reference import AdaptiveStreamingReference
from tests.research.joint_control.preview.test_causal_reference_replan import complete_prefix
from tests.research.joint_control.preview.test_streaming_feedback_boundary import fixture as feedback_fixture
from tests.research.native_runtime.test_adaptive_streaming_continuation import make_case
from tests.research.native_runtime.test_streaming_continuation import response_for


def setup_adapter(tmp_path, *, windows=3, max_periods=12, bad_capture=False, gate=None):
    case = make_case(tmp_path, windows=windows, max_periods=max_periods)
    events, owner = [], {"completed": 0}

    def render_anchor(_env, step, sample, endpoint, output):
        directory = output / f"adaptive_anchor{step:04d}"
        directory.mkdir()
        data = f"synthetic image at {step}".encode()
        path = directory / "current.png"
        path.write_bytes(data)
        before = {"sim_time": endpoint["sim_time"], "sim_state_sha256": endpoint["sim_state_sha256"]}
        after = dict(before)
        if bad_capture:
            after["sim_time"] += 0.002
        return {
            "frame_index": step,
            "current_endpoint": endpoint,
            "current_source_pose": sample["source_pose"],
            "current_opening": sample["opening"],
            "render_before": before,
            "render_after": after,
            "future_rgb_frames": 0,
            "camera_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "canvas_shape": [960, 640, 3],
            "canvas_dtype": "uint8",
            "canvas_png": {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()},
        }

    case.observers.render_anchor = render_anchor

    class Policy:
        def __init__(self, step, poses, openings, anchor):
            self.step, self.poses, self.openings, self.anchor = step, poses, openings, anchor
            self.pending = None

        def select(self, _q, _limits, _jacobian, previous, offset):
            step = self.step + offset
            assert self.pending is None and owner["completed"] == step - 80
            assert previous[0] == pytest.approx((step - 80) * 0.01)
            command = np.zeros(7)
            command[0] = (step - 79) * 0.01
            self.pending = (offset, command.copy())
            errors = np.zeros((8, 6))
            errors[1, 0] = 0.0095 if step + 1 in (83, 87) else 0.005
            return {
                "command": command,
                "newly_selected": True,
                "quality": {
                    "verification": {
                        "passed": True,
                        "first_pose_caps_satisfied": True,
                        "hard_pose_prefix": 2,
                        "prefix_pose_caps_satisfied": True,
                        "prefix_pose_caps_enforced": True,
                        "tcp_errors": errors,
                    }
                },
            }

        def complete(self, env, command, offset, actual_inner):
            assert env is case.env and len(actual_inner) == 25
            assert offset == self.pending[0] and np.array_equal(command, self.pending[1])
            self.pending = None
            owner["completed"] += 1
            return {"past_inner_error": {"last_completed_step": self.step + offset, "future_guarantee": False}}

    def first_policy(poses, openings, anchor, step):
        events.append(("initial", step, owner["completed"]))
        return Policy(step, poses, openings, anchor), owner

    def advance_policy(previous, actual_owner, poses, openings, anchor, *, generated_step, runtime_receipt):
        assert actual_owner is owner and previous.pending is None
        assert owner["completed"] == generated_step - 80
        assert runtime_receipt is case.receipt
        events.append(("advance", generated_step, owner["completed"]))
        return Policy(generated_step, poses, openings, anchor)

    simulation_socket, model_socket = socket.socketpair()
    channel = FramedJSONChannel(simulation_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    model_channel = FramedJSONChannel(model_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    adapter = AdaptiveStreamingReference(
        env=case.env,
        observers=case.observers,
        ledger=case.ledger,
        channel=channel,
        runtime_output=str(tmp_path.relative_to(repository_root())),
        tool=np.eye(4),
        receipt=case.receipt,
        first_policy=first_policy,
        response_gate=(lambda _request, _response: True) if gate is None else gate,
        advance_policy=advance_policy,
        score_gripper=lambda _env, opening: {"desired_opening": float(opening)},
    )
    return case, adapter, channel, model_channel, events, owner


def fake_service(case, channel, count, errors, *, stale=False, change_wait=False):
    try:
        for _ in range(count):
            request = channel.receive()
            inputs = case.service.admit(request)
            assert inputs == {
                "image_bytes": f"synthetic image at {request['step']}".encode(),
                "task": "place the object",
            }
            response = response_for(request)
            for row in response["action"]:
                row[3], row[7], row[9] = 1.0, 1.0, 0.7
            case.service.complete(request_id=request["request_id"], denoise_calls=35)
            if stale:
                response["request_id"] += 1
            if change_wait:
                case.sim.data.time += 0.002
            channel.send(response)
    except BaseException as error:
        errors.append(error)
    finally:
        channel.close()


def run_driver(case, adapter):
    return run_adaptive_streaming_continuation(
        case.env,
        case.terminal,
        case.observers,
        case.receipt,
        ledger=case.ledger,
        previous_command=np.zeros(7),
        tool=np.eye(4),
        target_object="target",
        open_segment=adapter,
        refresh_signal=completed_quality_margin,
        state_reader=case.state_reader,
        task_observer=case.task_observer,
    )


def test_current_images_and_history_drive_two_partial_replans(tmp_path):
    case, adapter, channel, model_channel, events, owner = setup_adapter(tmp_path)
    errors = []
    thread = threading.Thread(target=fake_service, args=(case, model_channel, 3, errors))
    try:
        thread.start()
        result = run_driver(case, adapter)
    finally:
        channel.close()
        thread.join(timeout=15)
    assert not thread.is_alive() and not errors
    assert result["complete"] and result["control_steps_completed"] == 12
    assert events == [("initial", 80, 0), ("advance", 83, 3), ("advance", 87, 7)]
    assert owner["completed"] == 12 and case.sim.inner == 300
    assert [item["frame_index"] for item in case.receipt["adaptive_captures"]] == [80, 83, 87]
    exchanges = case.receipt["adaptive_model_exchanges"]
    assert len(exchanges) == 3
    for step, exchange in zip((80, 83, 87), exchanges, strict=True):
        assert all(
            exchange[key]
            for key in (
                "request_send_attempted",
                "request_sent",
                "response_receive_attempted",
                "response_received",
                "response_accepted",
                "policy_ready",
            )
        )
        with open(exchange["request_ref"]["path"]) as stream:
            request = json.load(stream)
        with open(exchange["response_ref"]["path"]) as stream:
            response = json.load(stream)
        assert request["step"] == response["step"] == step
        assert request["image"]["sha256"] == response["image_sha256"]
    for row in result["segments"]:
        for period in row["continuation"]["periods"]:
            assert period["completed_feedback"]["score"]["desired_opening"] == pytest.approx(0.7)


@pytest.mark.parametrize("failure", ["gate", "stale", "change_wait"])
def test_bad_response_or_wait_is_saved_but_never_dispatched(tmp_path, failure):
    gate = (lambda _request, _response: False) if failure == "gate" else None
    case, adapter, channel, model_channel, events, _owner = setup_adapter(tmp_path, windows=1, gate=gate)
    errors = []
    thread = threading.Thread(
        target=fake_service,
        args=(case, model_channel, 1, errors),
        kwargs={"stale": failure == "stale", "change_wait": failure == "change_wait"},
    )
    try:
        thread.start()
        with pytest.raises(ValueError):
            run_driver(case, adapter)
    finally:
        channel.close()
        thread.join(timeout=15)
    assert not thread.is_alive() and not errors and not events
    exchange = case.receipt["adaptive_model_exchanges"][0]
    assert exchange["request_sent"] and exchange["response_received"] and not exchange["policy_ready"]
    assert exchange["response_accepted"] is (failure == "gate")
    assert case.receipt["adaptive_stream_partial"]["control_steps_attempted"] == 0
    assert case.ledger.phase == "failed"


def test_bad_capture_is_stopped_before_model_request(tmp_path):
    case, adapter, channel, model_channel, events, _owner = setup_adapter(tmp_path, windows=1, bad_capture=True)
    try:
        with pytest.raises(ValueError, match="Rendered cameras"):
            run_driver(case, adapter)
    finally:
        channel.close()
        model_channel.close()
    assert not events and case.ledger.issued_requests == 0 and case.sim.inner == 0
    assert case.ledger.phase == "failed"


def test_two_partial_boundaries_reuse_real_feedback_history_objects(tmp_path):
    case, adapter, channel, model_channel, _events, _owner = setup_adapter(tmp_path)
    policy, owner, _poses = feedback_fixture()
    policy.env = policy.arm_policy.env = case.env
    case.sim.data.time = 5.0
    old = case.ledger
    registration = {
        "session": old.session,
        "initial": measured_snapshot(case.env, 100),
        "image_root": old.image_root,
        "task": old.task,
        "seed": old.seed,
        "model_sha256": old.model_sha256,
        "max_requests": 3,
        "max_periods": 12,
    }
    case.ledger = adapter.ledger = AdaptiveStreamLedger(**registration)
    case.service = AdaptiveModelAdmission(**registration)
    policy.receipt.update(shared_shift_traces=[], causal_shift_fallback_traces=[])
    case.receipt["segment_receipts"] = [policy.receipt]
    case.receipt["segment_trace_starts"] = [{"shared": 0, "causal": 0}]
    adapter.first_policy = lambda _poses, _openings, _anchor, step: (policy, owner) if step == 100 else None
    adapter.advance_policy = advance_recorded_segment
    coupled, tracking, gripper = policy.arm_policy.coupled, policy.arm_policy.tracking, policy.gripper

    def observation():
        current = case.ledger.current
        sample, _endpoint = case.observers.endpoint(case.env, current["step"])
        return {"snapshot": current, "source_pose": sample["source_pose"], "opening": sample["opening"]}

    def completed_periods(count, error):
        for _ in range(count):
            before = case.ledger.current
            case.ledger.begin_period(before=before)
            case.env.step(np.zeros(7))
            after = measured_snapshot(case.env, before["step"] + 1)
            case.ledger.complete_period(before=before, after=after, inner_samples=25)
        complete_prefix(adapter.policy, count, error=error)
        assert case.ledger.refresh_early(
            trigger={"replan": True, "current_step": case.ledger.current["step"], "reason": "quality"}
        )

    errors = []
    thread = threading.Thread(target=fake_service, args=(case, model_channel, 3, errors))
    try:
        thread.start()
        first = adapter(observation(), None)
        completed_periods(3, 0.2)
        second = adapter(observation(), first)
        assert adapter.policy.arm_policy.coupled is coupled and adapter.policy.arm_policy.tracking is tracking
        assert adapter.policy.gripper is gripper and owner.total_snapshot()["completed_periods"] == 3
        completed_periods(4, 0.3)
        adapter(observation(), second)
    finally:
        channel.close()
        thread.join(timeout=15)
    assert not thread.is_alive() and not errors
    assert [capture["frame_index"] for capture in case.receipt["adaptive_captures"]] == [100, 103, 107]
    assert adapter.policy.arm_policy.coupled is coupled and adapter.policy.arm_policy.tracking is tracking
    assert adapter.policy.gripper is gripper and gripper.latch == 7
    assert adapter.policy.inner_history.snapshot()["total_completed_periods"] == 7
    assert adapter.policy.inner_history.snapshot()["inner_position_error_rad"][24, 0] == pytest.approx(0.3)
    assert len(case.receipt["segment_receipts"]) == len(case.receipt["segment_trace_starts"]) == 3
