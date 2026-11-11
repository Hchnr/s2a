"""Exercise capture, a fake model service and the existing continuation together."""

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading

import numpy as np
import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.repository_paths import repository_root
from s2a.research.joint_control.preview.streaming_feedback_boundary import advance_feedback
from s2a.research.native_runtime.streaming_continuation import (
    StreamingSegment,
    measured_snapshot,
    run_streaming_continuation,
)
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission, CausalStreamLedger
from s2a.research.wam.tracking.runtime.causal_stream_reference import CausalStreamingReference
from tests.research.joint_control.preview.test_streaming_feedback_boundary import complete_eight
from tests.research.joint_control.preview.test_streaming_feedback_boundary import fixture as feedback_fixture
from tests.research.native_runtime.test_streaming_continuation import fake_case, response_for


def setup_adapter(tmp_path, *, windows=2, bad_capture=False, gate=None):
    case = fake_case(tmp_path, windows=windows)
    root = repository_root()
    events = []
    owner = {"completed": 0}

    def render_anchor(_env, step, sample, endpoint, output):
        directory = output / f"anchor{step:04d}"
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
            assert self.pending is None and owner["completed"] == self.step - 80 + offset
            expected_previous = max(0, self.step - 80 + offset) * 0.01
            assert previous[0] == pytest.approx(expected_previous)
            command = np.zeros(7)
            command[0] = (self.step - 79 + offset) * 0.01
            self.pending = (offset, command.copy())
            return {"command": command, "newly_selected": True}

        def complete(self, env, command, offset, actual_inner):
            assert env is case.env and len(actual_inner) == 25
            assert offset == self.pending[0] and np.array_equal(command, self.pending[1])
            self.pending = None
            owner["completed"] += 1
            return {"completed": owner["completed"]}

    def first_policy(poses, openings, anchor, step):
        events.append(("initial", step, owner["completed"]))
        return Policy(step, poses, openings, anchor), owner

    def advance_policy(previous, actual_owner, poses, openings, anchor, *, generated_step, receipt):
        assert actual_owner is owner and previous.pending is None
        assert owner["completed"] == generated_step - 80
        events.append(("advance", generated_step, owner["completed"]))
        receipt["completed_prior"] = owner["completed"]
        return Policy(generated_step, poses, openings, anchor)

    simulation_socket, model_socket = socket.socketpair()
    channel = FramedJSONChannel(simulation_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    model_channel = FramedJSONChannel(model_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    adapter = CausalStreamingReference(
        env=case.env,
        observers=case.observers,
        ledger=case.ledger,
        channel=channel,
        runtime_output=str(tmp_path.relative_to(root)),
        tool=np.eye(4),
        receipt=case.receipt,
        first_policy=first_policy,
        response_gate=(lambda _request, _response: True) if gate is None else gate,
        advance_policy=advance_policy,
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
    return run_streaming_continuation(
        case.env,
        case.terminal,
        case.observers,
        case.receipt,
        ledger=case.ledger,
        max_windows=case.windows,
        previous_command=np.zeros(7),
        tool=np.eye(4),
        target_object="target",
        open_segment=adapter,
        state_reader=case.state_reader,
        task_observer=case.task_observer,
    )


def test_current_image_response_and_policy_history_drive_two_mock_windows(tmp_path):
    case, adapter, channel, model_channel, events, owner = setup_adapter(tmp_path)
    errors = []
    thread = threading.Thread(target=fake_service, args=(case, model_channel, 2, errors))
    try:
        thread.start()
        result = run_driver(case, adapter)
    finally:
        channel.close()
        thread.join(timeout=15)
    assert not thread.is_alive() and not errors
    assert result["complete"] and result["control_steps_completed"] == 16
    assert events == [("initial", 80, 0), ("advance", 88, 8)] and owner["completed"] == 16
    assert [item["frame_index"] for item in case.receipt["streaming_captures"]] == [80, 88]
    exchanges = case.receipt["streaming_model_exchanges"]
    assert len(exchanges) == 2
    for index, exchange in enumerate(exchanges):
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
        assert request["step"] == response["step"] == 80 + 8 * index
        assert request["image"]["sha256"] == response["image_sha256"]


@pytest.mark.parametrize("failure", ["gate", "stale", "change_wait"])
def test_rejected_response_is_saved_but_no_action_is_dispatched(tmp_path, failure):
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
    exchange = case.receipt["streaming_model_exchanges"][0]
    assert exchange["request_sent"] and exchange["response_received"] and not exchange["policy_ready"]
    assert exchange["response_accepted"] is (failure == "gate")
    assert case.receipt["streaming_partial"]["control_steps_attempted"] == 0
    assert case.ledger.phase == "failed"


def test_bad_capture_prevents_even_the_first_model_request(tmp_path):
    case, adapter, channel, model_channel, events, _owner = setup_adapter(tmp_path, windows=1, bad_capture=True)
    try:
        with pytest.raises(ValueError, match="Rendered cameras"):
            run_driver(case, adapter)
    finally:
        channel.close()
        model_channel.close()
    assert not events and case.ledger.issued_requests == 0
    assert case.receipt["streaming_model_exchanges"] == []


def test_send_failure_records_attempt_without_claiming_success(tmp_path):
    case, adapter, channel, model_channel, events, _owner = setup_adapter(tmp_path, windows=1)
    model_channel.close()
    try:
        with pytest.raises((BrokenPipeError, ConnectionResetError)):
            run_driver(case, adapter)
    finally:
        channel.close()
    exchange = case.receipt["streaming_model_exchanges"][0]
    assert exchange["request_send_attempted"] and not exchange["request_sent"]
    assert not exchange["response_receive_attempted"] and not events
    assert case.receipt["streaming_partial"]["control_steps_attempted"] == 0


def test_actual_feedback_boundary_accepts_the_next_generated_reference(tmp_path):
    case, adapter, channel, model_channel, _events, _owner = setup_adapter(tmp_path, windows=1)
    service = case.service
    config = {
        "session": service.session,
        "initial": measured_snapshot(case.env, 108),
        "image_root": service.image_root,
        "task": service.task,
        "seed": service.seed,
        "model_sha256": service.model_sha256,
        "max_requests": 1,
    }
    case.ledger = CausalStreamLedger(**config)
    case.service = CausalModelAdmission(**config)
    adapter.ledger = case.ledger
    policy, owner, _poses = feedback_fixture()
    complete_eight(policy, 0.2)
    policy.env = case.env
    adapter.policy, adapter.owner, adapter.advance_policy = policy, owner, advance_feedback
    previous = StreamingSegment(np.tile(np.eye(4), (8, 1, 1)), lambda *_args: None, lambda *_args: None)
    adapter.previous = previous
    current = {"snapshot": case.ledger.current, "source_pose": np.eye(4), "opening": 0.7}
    errors = []
    thread = threading.Thread(target=fake_service, args=(case, model_channel, 1, errors))
    try:
        thread.start()
        segment = adapter(current, previous)
    finally:
        channel.close()
        thread.join(timeout=15)
    assert not thread.is_alive() and not errors
    assert segment is adapter.previous and adapter.policy.start_step == 108
    assert adapter.policy.gripper is policy.gripper and policy.gripper.latch == 8
    assert adapter.policy.arm_policy.coupled is policy.arm_policy.coupled
    assert adapter.policy.inner_history.snapshot()["total_completed_periods"] == 8


def test_reference_driver_exchanges_two_windows_with_an_actual_model_child(tmp_path):
    case, adapter, channel, model_channel, events, owner = setup_adapter(tmp_path)
    service = case.service
    config = {
        "session": service.session,
        "initial": service.initial,
        "image_root": service.image_root,
        "task": service.task,
        "seed": service.seed,
        "model_sha256": service.model_sha256,
        "max_requests": service.max_requests,
    }
    code = """
import json, os, socket, sys
from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission
service = CausalModelAdmission(**json.loads(sys.argv[1]))
channel = FramedJSONChannel(socket.socket(fileno=int(sys.argv[2])), max_bytes=2*1024*1024, timeout_seconds=10)
for index in range(2):
    request = channel.receive()
    inputs = service.admit(request)
    assert inputs['image_bytes'] == f"synthetic image at {request['step']}".encode()
    identity = {key: request[key] for key in ('version','session','request_id','step','sim_time_s','state_sha256','seed','model_sha256')}
    action = [[0.0]*128 for _ in range(40)]
    for row in action: row[3], row[7], row[9] = 1.0, 1.0, 0.7
    response = dict(identity, image_sha256=request['image']['sha256'], denoise_calls=35,
                    action_dim_mask=[True]*10+[False]*118, action=action,
                    normalized_action=[[0.0]*128 for _ in range(40)])
    service.complete(request_id=index, denoise_calls=35)
    channel.send(response)
channel.close()
"""
    process = None
    try:
        descriptor = model_channel.connection.fileno()
        process = subprocess.Popen(
            [sys.executable, "-c", code, json.dumps(config), str(descriptor)],
            pass_fds=(descriptor,),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        model_channel.close()
        result = run_driver(case, adapter)
        stdout, _ = process.communicate(timeout=15)
        assert process.returncode == 0, stdout.decode(errors="replace")
    finally:
        channel.close()
        model_channel.close()
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=15)
    assert result["complete"] and result["control_steps_completed"] == 16
    assert events == [("initial", 80, 0), ("advance", 88, 8)] and owner["completed"] == 16
