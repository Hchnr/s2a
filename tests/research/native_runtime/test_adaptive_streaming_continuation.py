"""Exercise adaptive dispatch ordering against the real continuation runner."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.causal_reference_replan import completed_quality_margin
from s2a.research.native_runtime.adaptive_streaming_continuation import run_adaptive_streaming_continuation
from s2a.research.native_runtime.streaming_continuation import StreamingSegment, measured_snapshot
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveModelAdmission, AdaptiveStreamLedger
from tests.research.native_runtime.test_streaming_continuation import fake_case, response_for


def make_case(tmp_path, *, windows=4, max_periods=12, buffer_at=None, fail_at=None, render_changes=False):
    case = fake_case(tmp_path, windows=windows, buffer_at=buffer_at, fail_at=fail_at, render_changes=render_changes)
    old = case.ledger
    config = {
        "session": old.session,
        "initial": old.current,
        "image_root": old.image_root,
        "task": old.task,
        "seed": old.seed,
        "model_sha256": old.model_sha256,
        "max_requests": windows,
        "max_periods": max_periods,
    }
    case.ledger = AdaptiveStreamLedger(**config)
    case.service = AdaptiveModelAdmission(**config)
    case.trigger_steps = []
    return case


def execute(case, *, replan_steps=(), stop_at=None, change_wait=False, change_select=False, change_feedback=False):
    def open_segment(observation, previous):
        assert set(observation) == {"snapshot", "source_pose", "opening"}
        assert not observation["source_pose"].flags.writeable
        assert previous is (case.segments[-1] if case.segments else None)
        current = observation["snapshot"]
        request = case.ledger.issue(before_render=current, after_render=current, image=case.image)
        assert case.service.admit(request) == {"image_bytes": b"synthetic current image", "task": "place the object"}
        case.service.complete(request_id=request["request_id"], denoise_calls=35)
        if change_wait:
            case.sim.data.time += 0.002
        case.ledger.accept(response_for(request), after_wait=measured_snapshot(case.env, current["step"]))
        case.events.append(("request", current["step"], case.sim.inner))

        def select(_q, _limits, _jacobian, previous_command, offset):
            step = current["step"] + offset
            case.events.append(("select", step, case.sim.inner))
            if offset == 0:
                case.first_previous.append(previous_command.copy())
            if change_select:
                case.sim.data.time += 0.002
            if step == stop_at:
                return {"stop": {"reason": "quality_no_witness"}}
            command = np.zeros(7)
            command[0] = (step - 79) * 0.01
            errors = np.zeros((8, 6))
            errors[1, 0] = 0.0095 if step + 1 in replan_steps else 0.005
            verification = {
                "passed": True,
                "first_pose_caps_satisfied": True,
                "hard_pose_prefix": 2,
                "prefix_pose_caps_satisfied": True,
                "prefix_pose_caps_enforced": True,
                "tcp_errors": errors,
                "hidden_object_truth": "must not reach trigger",
            }
            return {"command": command, "newly_selected": True, "quality": {"verification": verification}}

        def complete(_env, command, offset, inner):
            step = current["step"] + offset
            assert len(inner) == 25 and command[0] == pytest.approx((step - 79) * 0.01)
            case.commits += 1
            case.events.append(("complete", step, case.sim.inner))
            if change_feedback:
                case.sim.data.time += 0.002
            return {
                "policy_history": {
                    "past_inner_error": {
                        "last_completed_step": step,
                        "future_guarantee": False,
                        "hidden_object_truth": "must not reach trigger",
                    }
                }
            }

        segment = StreamingSegment(np.tile(np.eye(4), (8, 1, 1)), select, complete)
        case.segments.append(segment)
        return segment

    def trigger(period, *, current_step):
        assert set(period) == {"step", "endpoint_receipt", "actual_inner", "decision", "completed_feedback"}
        assert set(period["endpoint_receipt"]) == {"endpoint_step"}
        assert "hidden_object_truth" not in period["decision"]["quality"]["verification"]
        assert "hidden_object_truth" not in period["completed_feedback"]["policy_history"]["past_inner_error"]
        case.trigger_steps.append(current_step)
        return completed_quality_margin(period, current_step=current_step)

    return run_adaptive_streaming_continuation(
        case.env,
        case.terminal,
        case.observers,
        case.receipt,
        ledger=case.ledger,
        previous_command=np.zeros(7),
        tool=np.eye(4),
        target_object="target",
        open_segment=open_segment,
        refresh_signal=trigger,
        state_reader=case.state_reader,
        task_observer=case.task_observer,
    )


def test_repeated_quality_replans_preserve_dispatch_feedback_and_previous_command(tmp_path):
    case = make_case(tmp_path)
    result = execute(case, replan_steps=(83, 87))
    assert result["complete"] and result["stop"]["reason"] == "period_budget_exhausted"
    assert result["requests_issued"] == result["responses_accepted"] == 3
    assert result["feedback_periods_committed"] == result["control_steps_completed"] == 12
    assert [row["start_step"] for row in result["segments"]] == [80, 83, 87]
    assert [row["continuation"]["steps_completed"] for row in result["segments"]] == [3, 4, 5]
    assert [row["boundary"] for row in result["segments"][:2]] == ["quality_trigger", "quality_trigger"]
    assert case.trigger_steps == [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
    np.testing.assert_allclose([command[0] for command in case.first_previous], [0.0, 0.03, 0.07])
    assert [event for event in case.events if event[0] == "request"] == [
        ("request", 80, 0),
        ("request", 83, 75),
        ("request", 87, 175),
    ]
    assert case.sim.inner == 300 and case.ledger.phase == "done"
    assert all(row["inflight"] is None for row in result["segments"])


def test_full_eight_period_window_and_absolute_physical_budget(tmp_path):
    case = make_case(tmp_path, max_periods=13)
    result = execute(case)
    assert result["complete"] and result["stop"]["reason"] == "period_budget_exhausted"
    assert [row["start_step"] for row in result["segments"]] == [80, 88]
    assert [row["continuation"]["steps_completed"] for row in result["segments"]] == [8, 5]
    assert result["segments"][0]["boundary"] == "window_end"
    assert result["feedback_periods_committed"] == 13 and case.sim.inner == 325


@pytest.mark.parametrize("replan_steps,expected", [((83,), 3), ((), 8)])
def test_request_budget_stops_before_any_further_dispatch(tmp_path, replan_steps, expected):
    case = make_case(tmp_path, windows=1, max_periods=12)
    result = execute(case, replan_steps=replan_steps)
    assert not result["complete"] and result["stop"]["reason"] == "request_budget_exhausted"
    assert result["requests_issued"] == 1
    assert result["feedback_periods_committed"] == result["control_steps_completed"] == expected
    assert case.sim.inner == expected * 25 and case.ledger.phase == "done"


def test_quality_stop_and_joint_buffer_loss_prevent_another_request(tmp_path):
    quality_path = tmp_path / "quality"
    quality_path.mkdir()
    case = make_case(quality_path)
    result = execute(case, stop_at=82)
    assert result["stop"]["reason"] == "quality_no_witness"
    assert result["requests_issued"] == 1 and result["feedback_periods_committed"] == 2
    assert case.sim.inner == 50 and case.ledger.phase == "failed"
    buffer_path = tmp_path / "buffer"
    buffer_path.mkdir()
    case = make_case(buffer_path, buffer_at=26)
    result = execute(case, replan_steps=(82,))
    assert result["stop"]["reason"] == "observed_joint_buffer_loss"
    assert result["requests_issued"] == 1 and result["feedback_periods_committed"] == 2
    assert case.sim.inner == 50 and case.ledger.phase == "failed"


def test_mid_period_failure_preserves_partial_dispatch_and_never_replans(tmp_path):
    case = make_case(tmp_path, fail_at=3)
    original_method = case.env.step
    with pytest.raises(RuntimeError, match="mid-period"):
        execute(case, replan_steps=(81,))
    partial = case.receipt["adaptive_stream_partial"]
    inflight = partial["segments"][0]["inflight"]
    assert partial["control_steps_attempted"] == 1 and partial["control_steps_completed"] == 0
    assert partial["feedback_periods_committed"] == 0 and partial["requests_issued"] == 1
    assert inflight["inner_attempted"] == inflight["inner_completed"] == len(inflight["actual_inner"]) == 3
    assert case.ledger.phase == "failed" and case.ledger.inflight["step"] == 80
    assert case.env.step == original_method and case.trigger_steps == []


def test_failed_feedback_retains_completed_dispatch_without_committing_it(tmp_path):
    case = make_case(tmp_path)
    with pytest.raises(ValueError, match="Feedback commit changed"):
        execute(case, change_feedback=True)
    partial = case.receipt["adaptive_stream_partial"]
    inflight = partial["segments"][0]["inflight"]
    assert partial["control_steps_attempted"] == partial["control_steps_completed"] == 1
    assert partial["feedback_periods_committed"] == 0 and partial["requests_issued"] == 1
    assert inflight["inner_attempted"] == inflight["inner_completed"] == len(inflight["actual_inner"]) == 25
    assert case.ledger.phase == "failed" and not case.trigger_steps


@pytest.mark.parametrize("failure", ["change_wait", "change_select", "render_changes"])
def test_state_changes_before_dispatch_fail_closed(tmp_path, failure):
    case = make_case(tmp_path, render_changes=failure == "render_changes")
    kwargs = {} if failure == "render_changes" else {failure: True}
    with pytest.raises(ValueError):
        execute(case, **kwargs)
    partial = case.receipt["adaptive_stream_partial"]
    assert partial["control_steps_attempted"] == partial["control_steps_completed"] == 0
    assert case.ledger.phase == "failed"


def test_absolute_600_period_budget_is_enforced_before_any_request(tmp_path):
    case = make_case(tmp_path, max_periods=521)
    with pytest.raises(ValueError, match="600-period"):
        execute(case)
    assert case.ledger.issued_requests == 0 and case.sim.inner == 0
