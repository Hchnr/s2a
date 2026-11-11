"""Count actual completed actions and reject altered multi-window receipts."""

import copy

import pytest

from scripts.simulation.runtime.reset_multitask_live_audit import summarize_branch


def period(step):
    return dict(
        step=step,
        actual_inner=[{}] * 25,
        tcp_error=[0.001, 0, 0, 0.01, 0, 0],
        pose_caps_satisfied=True,
        minimum_inner_margin_rad=0.5,
        target_grasp=step == 15,
        task_success=False,
    )


def producer():
    segment_receipt = dict(
        quality_refiner=dict(
            quality_response_evaluations=8,
            baseline_response_evaluations=0,
            endpoint_queries_charged=1600,
            transition_steps_charged=20800,
        ),
        gripper_proposals=dict(candidate_sequences=100),
    )
    first_receipt = dict(segment_receipt, shared_shift_traces=[{}] * 16, causal_shift_fallback_traces=[{}] * 16)
    stream = dict(
        segments=[
            dict(
                start_step=8,
                continuation=dict(periods=[period(i) for i in range(8, 16)], steps_completed=8),
                inflight=None,
            )
        ],
        stop=dict(reason="observed_joint_buffer_loss", at_step=16, next_action_dispatched=False),
        complete=False,
        automatic_retry=False,
        requests_issued=1,
        responses_accepted=1,
        control_steps_attempted=8,
        control_steps_completed=8,
        feedback_periods_committed=8,
    )
    runtime = dict(
        complete=True,
        passed=True,
        cleanup_complete=True,
        reset_observation_verified=True,
        reset_image_match=True,
        previous_saved_command=None,
        newly_selected_commands=16,
        control_steps_attempted=16,
        control_steps_completed=16,
        counters=dict(
            mj_step=dict(attempted=400, completed=400), model_construct=dict(completed=3), context=dict(completed=3)
        ),
        segment_receipts=[first_receipt, segment_receipt],
        segment_trace_starts=[dict(shared=0, causal=0), dict(shared=8, causal=8)],
        model_stop_sent=True,
        continuation=dict(first_window=dict(periods=[period(i) for i in range(8)]), stream=stream),
    )
    return dict(complete=True, passed=True, candidate_modules_unloaded=False, runtime=runtime)


def checker(source):
    result = copy.deepcopy(source)
    result["candidate_modules_unloaded"] = True
    runtime = result["runtime"]
    stream = runtime["continuation"].pop("stream")
    runtime["newly_selected_commands"] = 0
    runtime["continuation"].update(
        replays=[segment["continuation"] for segment in stream["segments"]],
        reported_producer_stop=stream["stop"],
        stop_reason_independently_verified=False,
        actual_model_requests=1,
        actual_stream_periods=8,
    )
    return result


def test_producer_and_checker_count_actual_periods_and_task_outcomes():
    source = producer()
    produced = summarize_branch(source, "producer")
    replayed = summarize_branch(checker(source), "checker", producer=source)
    assert produced["control_steps"] == replayed["control_steps"] == 16
    assert produced["native_steps"] == replayed["native_steps"] == 400
    assert produced["model_requests"] == 1 and replayed["model_requests"] == 0
    assert produced["target_grasp_periods"] == 1 and produced["task_success_periods"] == 0
    assert produced["position_rmse_m"] == pytest.approx(0.001)
    assert produced["rotation_rmse_rad"] == pytest.approx(0.01)


@pytest.mark.parametrize("alteration", ["extra_request", "changed_stop", "budget", "trace", "inner"])
def test_rejects_receipts_that_exceed_budget_or_launder_a_stop(alteration):
    source = producer()
    runtime = source["runtime"]
    stream = runtime["continuation"]["stream"]
    if alteration == "extra_request":
        stream["requests_issued"] += 1
    elif alteration == "changed_stop":
        stream["stop"]["at_step"] += 1
    elif alteration == "budget":
        runtime["segment_receipts"][1]["quality_refiner"]["endpoint_queries_charged"] = 9601
    elif alteration == "trace":
        runtime["segment_trace_starts"][1]["shared"] = 17
    else:
        stream["segments"][0]["continuation"]["periods"][0]["actual_inner"].pop()
    with pytest.raises(ValueError):
        summarize_branch(source, "producer")


def test_checker_cannot_claim_unexecuted_segments_or_recomputed_decision():
    source = producer()
    replayed = checker(source)
    replayed["runtime"]["continuation"]["stop_reason_independently_verified"] = True
    with pytest.raises(ValueError, match="isolated checker"):
        summarize_branch(replayed, "checker", producer=source)
