"""Connect a reproduced reset prefix to adaptive streaming and isolated replay."""

import copy
import socket
import threading

import numpy as np
import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.repository_paths import repository_root
from s2a.research.joint_control.preview.causal_reference_replan import completed_quality_margin
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveModelAdmission
from scripts.simulation.runtime import adaptive_live_bowl as adaptive
from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as isolation
from scripts.simulation.runtime import reset_bowl_next_reference as reference
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from tests.research.native_runtime.test_causal_replan_step107_execution import setup_capture
from tests.research.native_runtime.test_reset_bowl_next_reference import setup_next
from tests.research.native_runtime.test_reset_live_two_window import score, terminal_at
from tests.research.native_runtime.test_streaming_continuation import fake_case, response_for


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    monkeypatch.setattr(isolation.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(isolation.original, "checker_isolation", lambda: None)


def test_rebuilt_prefix_drives_two_live_references_and_independent_replay(tmp_path):
    prior, captured, (_raw, expected) = setup_capture(tmp_path)
    producer_dir = tmp_path / "adaptive_producer"
    producer_dir.mkdir()
    case, receipt, options, _events, first_reference = setup_next(producer_dir)
    factory, advance = options.pop("policy_factory"), options.pop("advance_policy")
    initial = {key: expected[key] for key in ("step", "sim_time_s", "state_sha256")}
    registration = dict(
        session="adaptive-bowl-synthetic",
        initial=initial,
        image_root=str(options["runtime_output"].relative_to(repository_root())),
        task="place the object",
        seed=42,
        model_sha256=expected["model_sha256"],
        max_requests=2,
        max_periods=8,
        denoise_steps=35,
    )
    service = AdaptiveModelAdmission(**registration)
    simulation_socket, model_socket = socket.socketpair()
    channel = FramedJSONChannel(simulation_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=15)
    model_channel = FramedJSONChannel(model_socket, max_bytes=2 * 1024 * 1024, timeout_seconds=15)
    errors, model_steps, switches = [], [], []

    class NewPolicy:
        def __init__(self, start, openings):
            self.start = self.completed = start
            self.openings = openings

        def select(self, _q, _limits, _jacobian, _previous, offset):
            assert self.completed - self.start == offset
            errors = np.zeros((8, 6))
            errors[1, 0] = 0.0095 if self.completed + 1 == 111 else 0.005
            return {
                "command": np.zeros(7),
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

        def complete(self, _env, _command, offset, inner):
            assert self.completed - self.start == offset and len(inner) == 25
            self.completed += 1
            return {"past_inner_error": {"last_completed_step": self.completed - 1, "future_guarantee": False}}

    def partial(previous, _owner, _poses, openings, _anchor, *, generated_step, runtime_receipt):
        assert previous.completed == generated_step and runtime_receipt is receipt
        switches.append(generated_step)
        return NewPolicy(generated_step, openings)

    def trigger(period, *, current_step):
        if current_step <= 107:
            return {"replan": current_step == 107, "reason": "quality", "current_step": current_step}
        return completed_quality_margin(period, current_step=current_step)

    def model():
        try:
            for _ in range(2):
                request = model_channel.receive()
                visible = service.admit(request)
                assert visible["image_bytes"] == f"current image at {request['step']}".encode()
                assert case.sim.inner == request["step"] * 25
                model_steps.append(request["step"])
                response = response_for(request)
                response["action"] = copy.deepcopy(expected["action"])
                response["normalized_action"] = copy.deepcopy(expected["normalized_action"])
                service.complete(request_id=request["request_id"], denoise_calls=35)
                model_channel.send(response)
        except BaseException as error:
            errors.append(error)
        finally:
            model_channel.close()

    worker = threading.Thread(target=model)
    try:
        worker.start()
        result = adaptive.run_producer(
            case.env,
            terminal_at(case, 0),
            case.observers,
            receipt,
            prefix_runner=reference.run_producer,
            prior_producer=prior,
            expected_replan_capture=captured,
            expected_response=expected,
            registration=registration,
            channel=channel,
            policy_factory=factory,
            advance_policy=advance,
            partial_advance=partial,
            trigger=trigger,
            **options,
        )
    finally:
        channel.close()
        worker.join(timeout=20)
    assert not worker.is_alive() and not errors
    assert model_steps == switches == [107, 111]
    assert receipt["control_steps_completed"] == 115 and case.sim.inner == 115 * 25
    assert [row["start_step"] for row in result["adaptive"]["segments"]] == [107, 111]
    assert result["adaptive"]["complete"] and result["adaptive"]["control_steps_completed"] == 8

    responses = [prefix_read(exchange["response_ref"]) for exchange in receipt["adaptive_model_exchanges"]]
    producer = {"runtime": {**receipt, "continuation": result}}
    checker_dir = tmp_path / "adaptive_checker"
    checker_dir.mkdir()
    checker = fake_case(checker_dir, windows=15)
    checker.observers.render_anchor = case.observers.render_anchor
    checker_receipt = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
    checker_registration = {**registration, "image_root": str(checker_dir.relative_to(repository_root()))}
    replayed = adaptive.run_checker(
        checker.env,
        terminal_at(checker, 0),
        checker.observers,
        checker_receipt,
        prefix_runner=reference.run_checker,
        producer=producer,
        expected_replan_capture=captured,
        responses=responses,
        registration=checker_registration,
        case=options["case"],
        expected_capture=options["expected_capture"],
        load_reference=lambda: first_reference,
        runtime_output=checker_dir,
        expected_second_capture=options["expected_second_capture"],
        load_second_reference=options["load_second_reference"],
        expected_step88_capture=options["expected_step88_capture"],
        load_step88_reference=options["load_step88_reference"],
        state_reader=checker.state_reader,
        task_observer=checker.task_observer,
        gripper_score=score,
        compare_score=feedback.compare_score,
        render_scope=lambda render, _scope, *args, **kwargs: render(*args, **kwargs),
    )
    assert checker.sim.inner == 115 * 25 and checker_receipt["newly_selected_commands"] == 0
    assert replayed["adaptive"]["complete"] and replayed["adaptive"]["control_steps_completed"] == 8
    assert not replayed["adaptive"]["stop_reason_independently_verified"]


def prefix_read(ref):
    from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

    return prefix.read(ref)


def test_first_reference_gate_rejects_stale_or_changed_responses():
    request = dict(request_id=0, step=107, sim_time_s=5.6, state_sha256="a" * 64, image={"sha256": "b" * 64})
    expected = dict(
        step=107,
        sim_time_s=5.6,
        state_sha256="a" * 64,
        image_sha256="b" * 64,
        model_sha256="c" * 64,
        seed=42,
        denoise_calls=35,
        action=[[0.0]],
        normalized_action=[[0.0]],
    )
    response = copy.deepcopy(expected)
    assert adaptive.match_first_reference(request, response, expected)
    response["action"][0][0] = 0.1
    with pytest.raises(AssertionError):
        adaptive.match_first_reference(request, response, expected)
    request["image"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="first live response"):
        adaptive.match_first_reference(request, expected, expected)
