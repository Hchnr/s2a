"""The CPU owner must count only dispatched fixed-reference periods."""

import copy
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime.run_reset_bowl_fixed_h40 import summarize_branch


def period(step):
    return dict(
        step=step,
        actual_inner=[{}] * 25,
        tcp_error=[0.001, 0, 0, 0.01, 0, 0],
        pose_caps_satisfied=True,
        minimum_inner_margin_rad=0.5,
        target_grasp=step == 15,
        task_success=False,
        completed_feedback=dict(score=dict(opening=0.9, opening_reference=0.8)),
    )


def producer():
    segment = dict(
        quality_refiner=dict(
            quality_response_evaluations=8,
            baseline_response_evaluations=0,
            endpoint_queries_charged=1600,
            transition_steps_charged=20800,
        ),
        gripper_proposals=dict(candidate_sequences=100),
    )
    first_receipt = dict(segment, shared_shift_traces=[{}] * 16, causal_shift_fallback_traces=[{}] * 16)
    fixed = dict(
        segments=[
            dict(
                start_step=8,
                reference_offset=0,
                generated_endpoints=40,
                terminal_hold_endpoints=0,
                continuation=dict(periods=[period(i) for i in range(8, 16)], steps_completed=8),
            )
        ],
        stop=dict(reason="observed_joint_buffer_loss", at_step=16, next_action_dispatched=False),
        complete=False,
        actual_model_requests=0,
        automatic_retry=False,
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
        segment_receipts=[first_receipt, segment],
        segment_trace_starts=[dict(shared=0, causal=0), dict(shared=8, causal=8)],
        continuation=dict(first_window=dict(periods=[period(i) for i in range(8)]), fixed=fixed),
    )
    return dict(complete=True, passed=True, runtime=runtime)


def test_summary_counts_completed_generated_targets_and_gripper_scores():
    result = summarize_branch(producer(), "producer")
    assert result["control_steps"] == 16 and result["native_steps"] == 400
    assert result["generated_reference_steps"] == 8 and result["last_generated_endpoint_index"] == 7
    assert result["model_requests"] == 0 and result["target_grasp_periods"] == 1
    assert result["minimum_reference_opening"] == 0.8 and result["minimum_actual_opening"] == 0.9


@pytest.mark.parametrize("alteration", ["offset", "hold", "stop", "budget", "inner"])
def test_summary_rejects_invented_targets_or_missing_work(alteration):
    branch = producer()
    runtime, fixed = branch["runtime"], branch["runtime"]["continuation"]["fixed"]
    if alteration == "offset":
        fixed["segments"][0]["reference_offset"] = 8
    elif alteration == "hold":
        fixed["segments"][0]["terminal_hold_endpoints"] = 8
    elif alteration == "stop":
        fixed["stop"]["at_step"] = 17
    elif alteration == "budget":
        runtime["segment_receipts"][1]["quality_refiner"]["endpoint_queries_charged"] = 9601
    else:
        fixed["segments"][0]["continuation"]["periods"][0]["actual_inner"].pop()
    with pytest.raises(ValueError):
        summarize_branch(branch, "producer")


def test_checker_must_replay_the_same_number_of_actual_actions():
    source = producer()
    checked = copy.deepcopy(source)
    checked["candidate_modules_unloaded"] = True
    runtime = checked["runtime"]
    fixed = runtime["continuation"].pop("fixed")
    runtime["newly_selected_commands"] = 0
    runtime["continuation"].update(
        replays=[row["continuation"] for row in fixed["segments"]],
        reported_producer_stop=fixed["stop"],
        stop_reason_independently_verified=False,
    )
    assert summarize_branch(checked, "checker", producer=source)["control_steps"] == 16
    runtime["continuation"]["replays"][0]["periods"].pop()
    with pytest.raises(ValueError, match="exact actual"):
        summarize_branch(checked, "checker", producer=source)


def test_owner_import_does_not_load_candidate_or_simulator():
    code = (
        "from scripts.simulation.runtime import run_reset_bowl_fixed_h40 as s; "
        "s.isolation.checker_isolation(); s.original.require_no_simulator_import()"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
