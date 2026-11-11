"""Owner accounting must preserve stops and never count undispatched periods."""

import copy
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import run_reset_live_second_window as owner


def fixture_branch(live_count=3):
    def period(i):
        return dict(step=i, actual_inner=[{}] * 25)

    segments = [
        dict(
            quality_refiner=dict(
                quality_response_evaluations=1,
                baseline_response_evaluations=0,
                endpoint_queries_charged=1,
                transition_steps_charged=1,
            ),
            gripper_proposals=dict(candidate_sequences=6561),
        )
        for _ in range(2)
    ]
    stop = dict(reason="quality_no_witness", at_step=8 + live_count, next_action_dispatched=False)
    stream = dict(
        segments=[dict(continuation=dict(periods=[period(i) for i in range(8, 8 + live_count)]))],
        requests_issued=1,
        responses_accepted=1,
        control_steps_attempted=live_count,
        control_steps_completed=live_count,
        feedback_periods_committed=live_count,
        complete=False,
        stop=stop,
    )
    counters = dict(
        mj_step=dict(attempted=25 * (8 + live_count), completed=25 * (8 + live_count)),
        model_construct=dict(completed=3),
        context=dict(completed=3),
    )
    runtime = dict(
        complete=True,
        passed=True,
        cleanup_complete=True,
        reset_observation_verified=True,
        reset_image_match=True,
        previous_saved_command=None,
        continuation=dict(first_window=dict(periods=[period(i) for i in range(8)]), stream=stream),
        counters=counters,
        control_steps_completed=8 + live_count,
        newly_selected_commands=8 + live_count,
        segment_receipts=segments,
    )
    return dict(complete=True, passed=True, runtime=runtime, candidate_modules_unloaded=False)


def test_producer_counts_only_actual_completed_periods_and_retains_stop():
    branch = fixture_branch()
    counts = owner.summarize_branch(branch, "producer")
    assert counts["control_steps"] == 11 and counts["native_steps"] == 275
    assert counts["model_requests"] == 1 and counts["stop"]["reason"] == "quality_no_witness"
    bad = copy.deepcopy(branch)
    bad["runtime"]["continuation"]["stream"]["stop"]["next_action_dispatched"] = True
    with pytest.raises(ValueError, match="explicitly stop"):
        owner.summarize_branch(bad, "producer")
    bad = copy.deepcopy(branch)
    bad["runtime"]["counters"]["mj_step"]["completed"] += 1
    with pytest.raises(ValueError, match="physical counts"):
        owner.summarize_branch(bad, "producer")


def test_importing_owner_does_not_load_candidate_or_simulator():
    code = "from scripts.simulation.runtime import run_reset_live_second_window as s; s.isolation.checker_isolation(); s.original.require_no_simulator_import()"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
