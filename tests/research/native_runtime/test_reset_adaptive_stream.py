"""Exercise reset-origin adaptive feedback with real stream and replay code."""

import copy
import hashlib
import socket
import threading

import numpy as np
import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.repository_paths import repository_root
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveModelAdmission, check_adaptive_stop
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_adaptive_stream as stream
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case, response_for


class Policy:
    def __init__(self, start=0, *, stop_at=None):
        self.start = self.completed = start
        self.openings = np.full(40, 0.7)
        self.stop_at = stop_at

    def select(self, _q, _limits, _jacobian, _previous, offset):
        assert self.completed == self.start + offset
        if self.completed == self.stop_at:
            return {"stop": {"reason": "quality_no_witness"}}
        return {
            "command": np.zeros(7),
            "newly_selected": True,
            "quality": {"verification": {"passed": True, "first_pose_caps_satisfied": True}},
        }

    def complete(self, _env, _command, _offset, inner):
        assert len(inner) == 25
        self.completed += 1
        return {"past_inner_error": {"last_completed_step": self.completed - 1, "future_guarantee": False}}


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


def first_window(tmp_path):
    case = fake_case(tmp_path, windows=1)
    receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    policy, poses = Policy(), np.tile(np.eye(4), (40, 1, 1))

    def complete(env, command, offset):
        inner = receipt["continuation_partial"]["periods"][-1]["actual_inner"]
        return dict(score=score(env, 0.7), policy_history=policy.complete(env, command, offset, inner))

    result = run_continuation(
        case.env,
        terminal_at(case, 0),
        case.observers,
        receipt,
        start_step=0,
        previous_command=np.zeros(7),
        targets=poses[:8],
        tool=np.eye(4),
        target_object="target",
        choose_action=policy.select,
        stop_on_outside=True,
        on_completed=complete,
        state_reader=case.state_reader,
        task_observer=case.task_observer,
    )
    return result["periods"], poses


@pytest.mark.parametrize("seed", [42, 43])
@pytest.mark.parametrize("stop_at", [None, 14])
def test_reset_to_live_replans_and_independent_replay(tmp_path, monkeypatch, seed, stop_at):
    monkeypatch.setattr(isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(feedback, "checker_isolation", lambda: None)
    for name in ("baseline", "producer", "checker"):
        (tmp_path / name).mkdir()
    baseline, poses = first_window(tmp_path / "baseline")
    endpoint = baseline[-1]["endpoint_receipt"]
    registration = dict(
        session=f"reset-adaptive-{seed}",
        initial=dict(step=8, sim_time_s=endpoint["sim_time"], state_sha256=endpoint["sim_state_sha256"]),
        image_root=str((tmp_path / "producer").relative_to(repository_root())),
        task="place the object",
        seed=seed,
        model_sha256="b" * 64,
        max_requests=4,
        max_periods=12,
        denoise_steps=35,
    )
    producer = fake_case(tmp_path / "producer", windows=1)
    producer.observers.render_anchor = render_anchor
    receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    case = dict(
        chunk=dict(effective_targets_tcp=poses, openings=np.full(40, 0.7)), tool=np.eye(4), target_object="target"
    )
    owner, first_policy, events = {}, Policy(), []

    def advance(previous, actual_owner, _poses, _openings, _anchor, *, generated_step, runtime_receipt):
        assert actual_owner is owner and previous.completed == generated_step and runtime_receipt is receipt
        events.append(("advance", generated_step))
        return Policy(generated_step, stop_at=stop_at)

    def trigger(period, *, current_step):
        assert period["step"] == current_step - 1
        return dict(replan=current_step in (11, 15), reason="synthetic_quality", current_step=current_step)

    service = AdaptiveModelAdmission(**registration)
    simulation_socket, model_socket = socket.socketpair()
    channel = FramedJSONChannel(simulation_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=15)
    model_channel = FramedJSONChannel(model_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=15)
    errors, requests = [], []

    def model():
        try:
            for _ in range(registration["max_requests"]):
                request = model_channel.receive()
                if request.get("kind") == "stop":
                    check_adaptive_stop(service, request)
                    events.append(("stop", request["next_step"]))
                    break
                visible = service.admit(request)
                assert visible["image_bytes"] == f"current image at {request['step']}".encode()
                assert producer.sim.inner == request["step"] * 25 and request["seed"] == seed
                requests.append(request["step"])
                response = response_for(request)
                for row in response["action"]:
                    row[3], row[7], row[9] = 1.0, 1.0, 0.7
                service.complete(request_id=request["request_id"], denoise_calls=35)
                model_channel.send(response)
        except BaseException as error:
            errors.append(error)
        finally:
            model_channel.close()

    worker = threading.Thread(target=model)
    try:
        worker.start()
        produced = stream.run_producer(
            producer.env,
            terminal_at(producer, 0),
            producer.observers,
            receipt,
            case=case,
            baseline=baseline,
            registration=registration,
            channel=channel,
            runtime_output=tmp_path / "producer",
            state_reader=producer.state_reader,
            task_observer=producer.task_observer,
            policy_factory=lambda *_args: (first_policy, owner, np.zeros(7)),
            first_advance=advance,
            next_advance=advance,
            trigger=trigger,
            gripper_score=score,
            render_scope=lambda render, _scope, *args, **kwargs: render(*args, **kwargs),
        )
    finally:
        channel.close()
        worker.join(timeout=20)
    assert not worker.is_alive() and not errors
    assert requests == ([8, 11, 15] if stop_at is None else [8, 11])
    assert [event for event in events if event[0] == "advance"] == [("advance", step) for step in requests]
    expected_total = 20 if stop_at is None else 14
    assert producer.sim.inner == expected_total * 25 and receipt["control_steps_completed"] == expected_total
    assert produced["adaptive"]["stop"]["reason"] == (
        "period_budget_exhausted" if stop_at is None else "quality_no_witness"
    )
    assert events[-1] == ("stop", expected_total)

    responses = [prefix.read(row["response_ref"]) for row in receipt["adaptive_model_exchanges"]]
    actual = {"runtime": {**receipt, "continuation": produced}}
    checker = fake_case(tmp_path / "checker", windows=1)
    checker.observers.render_anchor = render_anchor
    checked = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    checked_registration = copy.deepcopy(registration)
    checked_registration["image_root"] = str((tmp_path / "checker").relative_to(repository_root()))
    replayed = stream.run_checker(
        checker.env,
        terminal_at(checker, 0),
        checker.observers,
        checked,
        case=case,
        producer=actual,
        responses=responses,
        registration=checked_registration,
        runtime_output=tmp_path / "checker",
        state_reader=checker.state_reader,
        task_observer=checker.task_observer,
        gripper_score=score,
        render_scope=lambda render, _scope, *args, **kwargs: render(*args, **kwargs),
    )
    assert checker.sim.inner == expected_total * 25 and checked["newly_selected_commands"] == 0
    assert replayed["adaptive"]["control_steps_completed"] == expected_total - 8
    assert replayed["adaptive"]["reported_producer_stop"] == produced["adaptive"]["stop"]
    assert replayed["adaptive"]["stop_reason_independently_verified"] is False
