"""Exercise reset-to-current-image feedback with a synthetic simulator and model."""

import hashlib
import socket
import threading

import numpy as np

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.repository_paths import repository_root
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission
from scripts.simulation.runtime.reset_live_two_window import run_producer
from tests.research.native_runtime.test_streaming_continuation import fake_case, response_for


def score(_env, reference):
    return dict(
        opening=0.7,
        opening_reference=float(reference),
        opening_error=0.7 - float(reference),
        qpos=[0, 0],
        qvel=[0, 0],
        motor_target=[0, 0],
    )


class Policy:
    def __init__(self, completed=0):
        self.completed, self.openings = completed, np.full(40, 0.7)

    def select(self, _q, _limits, _jacobian, _previous, offset):
        assert self.completed % 8 == offset
        return {"command": np.zeros(7), "newly_selected": True}

    def complete(self, _env, _command, _offset, inner):
        assert len(inner) == 25
        self.completed += 1
        return {"completed": self.completed}


def terminal_at(case, step):
    sample, endpoint = case.observers.endpoint(case.env, step)
    return {"sample": sample, "endpoint_receipt": endpoint}


def test_real_first_window_precedes_current_image_request_and_preserves_history(tmp_path):
    root = repository_root()
    baseline_dir, live_dir = tmp_path / "baseline", tmp_path / "live"
    baseline_dir.mkdir()
    live_dir.mkdir()
    baseline_case = fake_case(baseline_dir, windows=1)
    baseline_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    baseline_policy = Policy()
    poses = np.tile(np.eye(4), (40, 1, 1))
    baseline = run_continuation(
        baseline_case.env,
        terminal_at(baseline_case, 0),
        baseline_case.observers,
        baseline_receipt,
        start_step=0,
        previous_command=np.zeros(7),
        targets=poses[:8],
        tool=np.eye(4),
        target_object="target",
        choose_action=baseline_policy.select,
        stop_on_outside=True,
        on_completed=lambda env, command, offset: {
            "score": score(env, 0.7),
            "policy_history": baseline_policy.complete(
                env,
                command,
                offset,
                [item["q"] for item in baseline_receipt["continuation_partial"]["periods"][-1]["actual_inner"]],
            ),
        },
        state_reader=baseline_case.state_reader,
        task_observer=baseline_case.task_observer,
    )
    final = baseline["periods"][-1]["endpoint_receipt"]
    case = fake_case(live_dir, windows=1)
    terminal = terminal_at(case, 0)
    receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    registration = dict(
        session="reset-two-window-test",
        initial={"step": 8, "sim_time_s": final["sim_time"], "state_sha256": final["sim_state_sha256"]},
        image_root=str(live_dir.relative_to(root)),
        task="place the object",
        seed=42,
        model_sha256="b" * 64,
        max_requests=1,
        denoise_steps=35,
    )
    service = CausalModelAdmission(**registration)
    owner, events = {}, []
    first_policy = Policy()

    def render_anchor(_env, step, sample, endpoint, output):
        folder = output / f"anchor{step:04d}"
        folder.mkdir()
        data = b"current image after eight real periods"
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
            canvas_png=dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()),
        )

    case.observers.render_anchor = render_anchor

    def advance(previous, actual_owner, _poses, _openings, _anchor, *, generated_step, runtime_receipt):
        assert previous is first_policy and actual_owner is owner and previous.completed == 8 and generated_step == 8
        events.append(("advance", case.sim.inner))
        return Policy(completed=8)

    simulation_socket, model_socket = socket.socketpair()
    channel = FramedJSONChannel(simulation_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=15)
    model_channel = FramedJSONChannel(model_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=15)
    errors = []

    def model():
        try:
            request = model_channel.receive()
            visible = service.admit(request)
            assert visible == {"image_bytes": b"current image after eight real periods", "task": "place the object"}
            assert case.sim.inner == 200 and request["step"] == 8
            events.append(("request", case.sim.inner))
            response = response_for(request)
            for row in response["action"]:
                row[3], row[7], row[9] = 1.0, 1.0, 0.7
            service.complete(request_id=0, denoise_calls=35)
            model_channel.send(response)
        except BaseException as error:
            errors.append(error)
        finally:
            model_channel.close()

    worker = threading.Thread(target=model)
    try:
        worker.start()
        result = run_producer(
            case.env,
            terminal,
            case.observers,
            receipt,
            case=dict(
                chunk=dict(effective_targets_tcp=poses, openings=np.full(40, 0.7)),
                tool=np.eye(4),
                target_object="target",
            ),
            baseline=baseline["periods"],
            registration=registration,
            channel=channel,
            runtime_output=live_dir,
            state_reader=case.state_reader,
            task_observer=case.task_observer,
            policy_factory=lambda *_args: (first_policy, owner, np.zeros(7)),
            advance_policy=advance,
            gripper_score=score,
            render_scope=lambda render, scope, *args, **kwargs: render(*args, **kwargs),
        )
    finally:
        channel.close()
        worker.join(timeout=20)
    assert not worker.is_alive() and not errors
    assert result["first_window"]["steps_completed"] == 8
    assert result["stream"]["complete"] is True and result["stream"]["control_steps_completed"] == 8
    assert events == [("request", 200), ("advance", 200)]
    assert case.sim.inner == 400 and receipt["newly_selected_commands"] == 16
    assert len(receipt["streaming_captures"]) == len(receipt["streaming_model_exchanges"]) == 1
