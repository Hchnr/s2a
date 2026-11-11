"""Exercise the real eight-period executor with a synthetic simulator and service."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.infrastructure.repository_paths import repository_root
from s2a.research.native_runtime.streaming_continuation import (
    StreamingSegment,
    measured_snapshot,
    run_streaming_continuation,
)
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission, CausalStreamLedger


def fake_case(tmp_path, *, windows=3, buffer_at=None, fail_at=None, render_changes=False):
    class Sim:
        def __init__(self):
            self.data = SimpleNamespace(time=4.0, qpos=np.zeros(7), qvel=np.zeros(7))
            self.inner = 0

        def step(self):
            self.inner += 1
            self.data.time += 0.002
            if buffer_at is not None and self.inner >= buffer_at:
                self.data.qpos[0] = 0.96

        def get_state(self):
            state = np.zeros(79, dtype=np.float64)
            state[0] = self.data.time
            state[1:8] = self.data.qpos
            state[8:15] = self.data.qvel
            return state

    sim = Sim()
    robot = SimpleNamespace(_ref_joint_pos_indexes=np.arange(7), _ref_joint_vel_indexes=np.arange(7))

    class Env:
        robots = [robot]

        def __init__(self):
            self.sim = sim

        def step(self, _command):
            for _ in range(25):
                self.sim.step()
                if fail_at is not None and self.sim.inner == fail_at:
                    raise RuntimeError("synthetic mid-period failure")

    env = Env()
    limits = np.tile([-1.0, 1.0], (7, 1))

    def endpoint(_env, step):
        current = measured_snapshot(env, step)
        sample = {
            "sim_states": sim.get_state(),
            "source_pose": np.eye(4),
            "opening": 0.7,
            "native_success": True,
            "object_privileged_state": 99,
        }
        receipt = {
            "endpoint_step": step,
            "sim_time": current["sim_time_s"],
            "sim_state_sha256": current["state_sha256"],
            "native_success": True,
        }
        return sample, receipt

    def render(_env, _size):
        if render_changes:
            sim.data.time += 0.002

    observers = SimpleNamespace(endpoint=endpoint, current=lambda _env: None, render_cameras=render)

    def state_reader(_env):
        return sim.data.qpos.copy(), limits.copy(), np.zeros((6, 7))

    def task_observer(_env):
        return {"success": False, "objects": {"target": {"grasp_contact": False}}}

    initial = measured_snapshot(env, 80)
    sample, endpoint_receipt = endpoint(env, 80)
    prefix = [{"time": 4.0 - (24 - i) * 0.002, "q": np.zeros(7), "qvel": np.zeros(7)} for i in range(25)]
    receipt = {
        "control_steps_attempted": 80,
        "control_steps_completed": 80,
        "newly_selected_commands": 0,
        "final_prefix_inner": {
            "prefix_step": 79,
            "samples": prefix,
            "limits": limits.copy(),
            "below_buffer_samples": 0,
            "outside_samples": 0,
        },
    }
    image_data = b"synthetic current image"
    (tmp_path / "frame.png").write_bytes(image_data)
    image = {"path": "frame.png", "sha256": hashlib.sha256(image_data).hexdigest()}
    config = {
        "session": "streaming-fake-simulator",
        "initial": initial,
        "image_root": str(tmp_path.relative_to(repository_root())),
        "task": "place the object",
        "seed": 42,
        "model_sha256": "b" * 64,
        "max_requests": windows,
    }
    return SimpleNamespace(
        env=env,
        sim=sim,
        terminal={"sample": sample, "endpoint_receipt": endpoint_receipt},
        observers=observers,
        state_reader=state_reader,
        task_observer=task_observer,
        receipt=receipt,
        ledger=CausalStreamLedger(**config),
        service=CausalModelAdmission(**config),
        image=image,
        events=[],
        segments=[],
        first_previous=[],
        commits=0,
        windows=windows,
    )


def response_for(request):
    keys = ("version", "session", "request_id", "step", "sim_time_s", "state_sha256", "seed", "model_sha256")
    return {
        **{key: request[key] for key in keys},
        "image_sha256": request["image"]["sha256"],
        "denoise_calls": 35,
        "action_dim_mask": [True] * 10 + [False] * 118,
        "action": [[0.0] * 128 for _ in range(40)],
        "normalized_action": [[0.0] * 128 for _ in range(40)],
    }


def execute(case, *, stop_at=None, change_wait=False, change_select=False, change_feedback=False, bad_response=False):
    def open_segment(observation, previous):
        assert set(observation) == {"snapshot", "source_pose", "opening"}
        assert not observation["source_pose"].flags.writeable
        assert previous is (case.segments[-1] if case.segments else None)
        assert case.commits == len(case.segments) * 8
        current = observation["snapshot"]
        request = case.ledger.issue(before_render=current, after_render=current, image=case.image)
        assert case.service.admit(request) == {"image_bytes": b"synthetic current image", "task": "place the object"}
        case.service.complete(request_id=request["request_id"], denoise_calls=35)
        response = response_for(request)
        if bad_response:
            response["request_id"] += 1
        if change_wait:
            case.sim.data.time += 0.002
        case.ledger.accept(response, after_wait=measured_snapshot(case.env, current["step"]))
        case.events.append(("request", current["step"], case.sim.inner))

        def select(_q, _limits, _jacobian, previous_command, offset):
            case.events.append(("select", current["step"] + offset, case.sim.inner))
            if offset == 0:
                case.first_previous.append(previous_command.copy())
            if change_select:
                case.sim.data.time += 0.002
            if current["step"] + offset == stop_at:
                return {"stop": {"reason": "quality_no_witness"}}
            command = np.zeros(7)
            command[0] = (current["step"] + offset - 79) * 0.01
            return {"command": command, "newly_selected": True}

        def complete(_env, command, offset, inner):
            expected = np.zeros(7)
            expected[0] = (current["step"] + offset - 79) * 0.01
            assert len(inner) == 25 and np.array_equal(command, expected)
            case.commits += 1
            case.events.append(("complete", current["step"] + offset, case.sim.inner))
            if change_feedback:
                case.sim.data.time += 0.002
            return {"history_count": case.commits}

        segment = StreamingSegment(np.tile(np.eye(4), (8, 1, 1)), select, complete)
        case.segments.append(segment)
        return segment

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
        open_segment=open_segment,
        state_reader=case.state_reader,
        task_observer=case.task_observer,
    )


def test_three_windows_preserve_completed_feedback_and_actual_period_order(tmp_path):
    case = fake_case(tmp_path)
    result = execute(case)
    assert result["complete"] and result["stop"] is None
    assert result["requests_issued"] == result["responses_accepted"] == 3
    assert result["feedback_periods_committed"] == result["control_steps_completed"] == 24
    assert case.ledger.phase == case.service.phase == "done"
    np.testing.assert_allclose([command[0] for command in case.first_previous], [0.0, 0.08, 0.16])
    assert [event for event in case.events if event[0] == "request"] == [
        ("request", 80, 0),
        ("request", 88, 200),
        ("request", 96, 400),
    ]
    for index, row in enumerate(result["segments"]):
        assert row["start_step"] == 80 + 8 * index
        assert row["continuation"]["steps_completed"] == 8
        assert row["inflight"] is None
    assert case.sim.inner == 600


def test_no_witness_stops_before_the_next_dispatch_or_request(tmp_path):
    case = fake_case(tmp_path)
    result = execute(case, stop_at=82)
    assert not result["complete"] and result["stop"]["reason"] == "quality_no_witness"
    assert result["requests_issued"] == 1
    assert result["control_steps_attempted"] == result["feedback_periods_committed"] == 2
    assert case.sim.inner == 50 and case.ledger.phase == "failed"


@pytest.mark.parametrize(("buffer_at", "expected_periods"), [(26, 2), (176, 8)])
def test_inner_buffer_loss_stops_before_another_dispatch_or_request(tmp_path, buffer_at, expected_periods):
    case = fake_case(tmp_path, buffer_at=buffer_at)
    result = execute(case)
    assert not result["complete"] and result["stop"]["reason"] == "observed_joint_buffer_loss"
    assert result["requests_issued"] == 1
    assert result["control_steps_completed"] == result["feedback_periods_committed"] == expected_periods
    assert case.sim.inner == expected_periods * 25


def test_prefix_buffer_loss_prevents_any_model_request(tmp_path):
    case = fake_case(tmp_path)
    inner = case.receipt["final_prefix_inner"]
    inner["samples"][0]["q"][0] = 0.96
    inner["below_buffer_samples"] = 1
    result = execute(case)
    assert result["stop"]["reason"] == "observed_prefix_joint_buffer_loss"
    assert result["requests_issued"] == result["control_steps_attempted"] == 0


@pytest.mark.parametrize("failure", ["change_wait", "change_select", "bad_response", "render_changes"])
def test_invalid_or_state_changing_reference_pipeline_never_dispatches(tmp_path, failure):
    case = fake_case(tmp_path, render_changes=failure == "render_changes")
    kwargs = {} if failure == "render_changes" else {failure: True}
    with pytest.raises(ValueError):
        execute(case, **kwargs)
    partial = case.receipt["streaming_partial"]
    assert partial["control_steps_attempted"] == partial["control_steps_completed"] == 0
    assert case.ledger.phase == "failed"


def test_failed_feedback_keeps_completed_dispatch_and_all_inner_samples(tmp_path):
    case = fake_case(tmp_path)
    with pytest.raises(ValueError, match="Feedback commit changed"):
        execute(case, change_feedback=True)
    partial = case.receipt["streaming_partial"]
    inflight = partial["segments"][0]["inflight"]
    assert partial["control_steps_attempted"] == partial["control_steps_completed"] == 1
    assert partial["feedback_periods_committed"] == 0
    assert inflight["inner_attempted"] == inflight["inner_completed"] == len(inflight["actual_inner"]) == 25
    assert case.ledger.phase == "failed"


def test_mid_period_failure_retains_dispatched_command_and_partial_inner_samples(tmp_path):
    case = fake_case(tmp_path, fail_at=3)
    with pytest.raises(RuntimeError, match="mid-period"):
        execute(case)
    partial = case.receipt["streaming_partial"]
    inflight = partial["segments"][0]["inflight"]
    assert partial["control_steps_attempted"] == 1 and partial["control_steps_completed"] == 0
    assert partial["feedback_periods_committed"] == 0
    assert inflight["step"] == 80 and inflight["command"][0] == 0.01
    assert inflight["inner_attempted"] == inflight["inner_completed"] == len(inflight["actual_inner"]) == 3
    assert case.ledger.phase == "failed"
