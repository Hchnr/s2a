"""A reset policy must retain causal history through multiple new references."""

import hashlib
import socket
import threading

import numpy as np
import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.repository_paths import repository_root
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.wam.tracking.runtime.causal_stream_control import check_stop
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission
from scripts.simulation.runtime.reset_live_multi_window import advance_registered_segment, run_producer
from tests.research.joint_control.preview.test_streaming_feedback_boundary import complete_eight
from tests.research.joint_control.preview.test_streaming_feedback_boundary import fixture as real_feedback_fixture
from tests.research.native_runtime.test_reset_live_two_window import Policy, score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case, response_for


def test_real_segment_boundary_preserves_shared_trace_offsets():
    policy, owner, poses = real_feedback_fixture()
    complete_eight(policy, 0.2)
    policy.receipt["shared_shift_traces"] = [{}, {}]
    policy.receipt["causal_shift_fallback_traces"] = [{}]
    receipt = dict(segment_receipts=[policy.receipt], segment_trace_starts=[dict(shared=0, causal=0)])
    segment_receipt = {}
    next_policy = advance_registered_segment(
        policy,
        owner,
        poses,
        np.ones(40),
        np.eye(4),
        generated_step=108,
        segment_receipt=segment_receipt,
        runtime_receipt=receipt,
    )
    assert next_policy.inner_history.snapshot()["total_completed_periods"] == 8
    assert receipt["segment_trace_starts"][-1] == dict(shared=2, causal=1)
    assert next_policy.receipt is segment_receipt and len(receipt["segment_receipts"]) == 1
    with pytest.raises(ValueError, match="latest history"):
        advance_registered_segment(
            next_policy,
            owner,
            poses,
            np.ones(40),
            np.eye(4),
            generated_step=116,
            segment_receipt={},
            runtime_receipt=receipt,
        )


def execute_fake(tmp_path, *, stop_at=None, buffer_at=None, include_context=False):
    root = repository_root()
    baseline_dir, live_dir = tmp_path / "baseline", tmp_path / "live"
    baseline_dir.mkdir()
    live_dir.mkdir()
    reference_case = fake_case(baseline_dir, windows=1, buffer_at=buffer_at)
    baseline_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    initial_policy = Policy()
    poses = np.tile(np.eye(4), (40, 1, 1))

    def baseline_complete(env, command, offset):
        period = baseline_receipt["continuation_partial"]["periods"][-1]
        return dict(
            score=score(env, 0.7),
            policy_history=initial_policy.complete(
                env, command, offset, [item["q"] for item in period["actual_inner"]]
            ),
        )

    baseline = run_continuation(
        reference_case.env,
        terminal_at(reference_case, 0),
        reference_case.observers,
        baseline_receipt,
        start_step=0,
        previous_command=np.zeros(7),
        targets=poses[:8],
        tool=np.eye(4),
        target_object="target",
        choose_action=initial_policy.select,
        stop_on_outside=True,
        on_completed=baseline_complete,
        state_reader=reference_case.state_reader,
        task_observer=reference_case.task_observer,
    )
    final = baseline["periods"][-1]["endpoint_receipt"]
    case = fake_case(live_dir, windows=5, buffer_at=buffer_at)
    terminal = terminal_at(case, 0)
    receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    registration = dict(
        session="reset-multi-test",
        initial={"step": 8, "sim_time_s": final["sim_time"], "state_sha256": final["sim_state_sha256"]},
        image_root=str(live_dir.relative_to(root)),
        task="place the object",
        seed=42,
        model_sha256="b" * 64,
        max_requests=5,
        denoise_steps=35,
    )
    service = CausalModelAdmission(**registration)
    history_owner, events = {}, []
    first_policy = Policy()

    class LivePolicy(Policy):
        def select(self, q, limits, jacobian, previous, offset):
            if self.completed == stop_at:
                return {"stop": {"reason": "quality_no_witness"}}
            return super().select(q, limits, jacobian, previous, offset)

    def render_anchor(_env, step, sample, endpoint, output):
        folder = output / f"anchor{step:04d}"
        folder.mkdir()
        data = f"current image at {step}".encode()
        path = folder / "current.png"
        path.write_bytes(data)
        binding = {"sim_time": endpoint["sim_time"], "sim_state_sha256": endpoint["sim_state_sha256"]}
        return dict(
            frame_index=step,
            current_endpoint=endpoint,
            current_source_pose=sample["source_pose"],
            current_opening=sample["opening"],
            render_before=binding,
            render_after=dict(binding),
            future_rgb_frames=0,
            camera_keys=["agentview_image", "robot0_eye_in_hand_image"],
            canvas_shape=[960, 640, 3],
            canvas_dtype="uint8",
            canvas_rgb_sha256=hashlib.sha256(data + b" rgb").hexdigest(),
            canvas_png=dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()),
        )

    case.observers.render_anchor = render_anchor

    def first_advance(previous, actual_owner, _poses, _openings, _anchor, *, generated_step, runtime_receipt):
        assert previous is first_policy and actual_owner is history_owner and previous.completed == generated_step == 8
        events.append(("advance", generated_step, case.sim.inner))
        return LivePolicy(completed=generated_step)

    def next_advance(
        previous, actual_owner, _poses, _openings, _anchor, *, generated_step, segment_receipt, runtime_receipt
    ):
        assert actual_owner is history_owner and previous.completed == generated_step
        events.append(("advance", generated_step, case.sim.inner))
        segment_receipt["completed_prior"] = generated_step
        return LivePolicy(completed=generated_step)

    simulation_socket, model_socket = socket.socketpair()
    channel = FramedJSONChannel(simulation_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=15)
    model_channel = FramedJSONChannel(model_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=15)
    errors = []

    def model():
        try:
            for _ in range(5):
                request = model_channel.receive()
                if request.get("kind") == "stop":
                    events.append(("stop", check_stop(service, request)["next_step"], case.sim.inner))
                    break
                visible = service.admit(request)
                assert visible == {
                    "image_bytes": f"current image at {request['step']}".encode(),
                    "task": "place the object",
                }
                assert case.sim.inner == 25 * request["step"]
                events.append(("request", request["step"], case.sim.inner))
                response = response_for(request)
                for row in response["action"]:
                    row[3], row[7], row[9] = 1.0, 1.0, 0.7
                service.complete(request_id=request["request_id"], denoise_calls=35)
                model_channel.send(response)
        except BaseException as error:
            errors.append(error)
        finally:
            model_channel.close()

    case_input = dict(
        chunk=dict(effective_targets_tcp=poses, openings=np.full(40, 0.7)),
        tool=np.eye(4),
        target_object="target",
    )
    worker = threading.Thread(target=model)
    try:
        worker.start()
        result = run_producer(
            case.env,
            terminal,
            case.observers,
            receipt,
            case=case_input,
            baseline=baseline["periods"],
            registration=registration,
            channel=channel,
            runtime_output=live_dir,
            state_reader=case.state_reader,
            task_observer=case.task_observer,
            policy_factory=lambda *_args: (first_policy, history_owner, np.zeros(7)),
            first_advance=first_advance,
            next_advance=next_advance,
            gripper_score=score,
            render_scope=lambda render, _scope, *args, **kwargs: render(*args, **kwargs),
        )
    finally:
        channel.close()
        worker.join(timeout=20)
    assert not worker.is_alive() and not errors
    if include_context:
        return result, receipt, case, events, dict(registration=registration, case=case_input, render=render_anchor)
    return result, receipt, case, events


def test_five_live_windows_follow_completed_reset_history(tmp_path):
    result, receipt, case, events = execute_fake(tmp_path)
    stream = result["stream"]
    assert result["first_window"]["steps_completed"] == 8 and stream["complete"] is True
    assert stream["requests_issued"] == 5 and stream["control_steps_completed"] == 40
    assert [event[1] for event in events if event[0] == "request"] == [8, 16, 24, 32, 40]
    assert [event[1] for event in events if event[0] == "advance"] == [8, 16, 24, 32, 40]
    assert case.sim.inner == 1200 and receipt["newly_selected_commands"] == 48


def test_mid_window_stop_closes_waiting_model_without_more_actions(tmp_path):
    result, receipt, case, events = execute_fake(tmp_path, stop_at=19)
    stream = result["stream"]
    assert stream["complete"] is False and stream["stop"]["reason"] == "quality_no_witness"
    assert stream["requests_issued"] == 2 and stream["control_steps_completed"] == 11
    assert events[-1] == ("stop", 19, 475)
    assert case.sim.inner == 475 and receipt["newly_selected_commands"] == 19
