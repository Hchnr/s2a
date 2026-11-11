"""Verify strict-prefix parity, actual-buffer gating, and full cost accounting."""

import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_model_safe_tracking_fallback as binding
from scripts.simulation.runtime import run_model_safe_tracking_fallback as runner


def test_strict_prefix_is_checked_before_and_not_after_first_fallback(monkeypatch):
    comparisons = []
    monkeypatch.setattr(runner.long, "check_previous_period", lambda actual, prior: comparisons.append((actual, prior)))
    previous = [{"id": "strict"}]
    state = {"strict_periods_verified": 0, "fallback_seen": False}
    runner.verify_strict_period({"id": "actual"}, previous, 0, "quality_witness", state)
    runner.verify_strict_period({}, previous, 1, runner.DEGRADED, state)
    runner.verify_strict_period({}, previous, 2, "quality_witness", state)
    assert comparisons == [({"id": "actual"}, {"id": "strict"})]
    assert state == {"strict_periods_verified": 1, "fallback_seen": True}
    prior = {
        "runtime": {"continuation": {"periods": previous, "stop": {"reason": "quality_no_witness", "at_step": 17}}}
    }
    runner.verify_strict_prefix_closed({"periods": [{}, {}], "stop": None}, prior, state)


def test_changed_or_truncated_strict_prefix_is_rejected():
    previous = [{"id": "strict"}]
    state = {"strict_periods_verified": 0, "fallback_seen": False}
    with pytest.raises(ValueError, match="exactly"):
        runner.verify_strict_period({}, previous, 0, runner.DEGRADED, state)
    with pytest.raises(ValueError, match="changed strict continuation"):
        runner.verify_strict_period({}, previous, 1, "quality_witness", state)
    prior = {"runtime": {"continuation": {"periods": previous, "stop": None}}}
    with pytest.raises(ValueError, match="complete strict prefix"):
        runner.verify_strict_prefix_closed({"periods": [], "stop": None}, prior, state)


def test_independent_audit_rejection_can_stop_at_the_original_boundary():
    state = {"strict_periods_verified": 1, "fallback_seen": False}
    prior = {"runtime": {"continuation": {"periods": [{}], "stop": {"reason": "quality_no_witness", "at_step": 17}}}}
    outcome = {"periods": [{}], "stop": {"reason": "quality_witness_rejected", "at_step": 17}}
    runner.verify_strict_prefix_closed(outcome, prior, state)


def test_actual_inner_buffer_loss_blocks_next_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.builder, "build_refiner", lambda *_args: (object(), object()))

    def callbacks(_policy, _history, _receipt, _start):
        def select(_q, _limits, _jacobian, _previous, offset):
            calls.append(("select", offset))
            return {"command": np.zeros(7), "newly_selected": True}

        return select, lambda _env, _command, offset: {"observed": offset}

    monkeypatch.setattr(runner.quality, "quality_callbacks", callbacks)
    q, limits, jac, previous = np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7)), np.zeros(7)
    receipt = {"final_prefix_inner": {"below_buffer_samples": 0}, "continuation_partial": {"periods": []}}
    action, completed = runner.feedback_callbacks(None, "task2", {"spec": {"prefix_steps": 32}}, {}, receipt)
    assert "command" in action(q, limits, jac, previous, 0)
    samples = np.zeros((25, 7))
    samples[4, 1] = 0.96
    receipt["continuation_partial"]["periods"].append({"actual_inner": [{"q": row} for row in samples]})
    assert completed(None, previous, 0)["completed_buffer_gate"]["below_buffer"]
    assert action(q, limits, jac, previous, 1)["stop"]["reason"] == "observed_joint_buffer_loss"
    assert calls == [("select", 0)]


def simulated_branch(index, scope):
    degraded = scope == runner.DEGRADED
    strict = scope == "strict"
    selected = degraded or strict
    old_count = 2
    record = {
        "strict_responses": old_count,
        "extra_rollouts": int(degraded),
        "response_evaluations": old_count + int(degraded),
        "selected_scope": scope,
        "audit_status": "passed" if selected else "not_attempted",
    }
    prior = {
        "response_evaluations": old_count,
        "fallback_rollouts": int(degraded),
        "audit_status": "passed" if strict else "not_attempted",
        "fallback_safety": {"accepted": True},
        "fallback_first_pose_caps_satisfied": False,
        "primary_initialization": {"start_lp_attempts": 1},
    }
    runtime = {
        "cleanup_complete": True,
        "control_steps_completed": 20,
        "counters": {"mj_step": {"completed": 500}, "model_construct": {"completed": 3}, "context": {"completed": 3}},
        "newly_selected_commands": int(selected),
        "refiner_calls": {"attempted": 1},
        "quality_refiner": {
            "baseline_response_evaluations": 3,
            "quality_calls_attempted": 1,
            "quality_calls_completed": 1,
            "quality_response_evaluations": record["response_evaluations"],
            "endpoint_queries_charged": 200 if selected else 0,
            "transition_steps_charged": 2600 if selected else 0,
            "witness_audits_passed": int(selected),
        },
        "model_safe_tracking_fallback_traces": [record],
        "causal_shift_fallback_traces": [prior],
        "shared_shift_traces": [{"response_evaluations": old_count}],
    }
    return {"case": {"case_index": index}, "branch": runner.MODE, "complete": True, "passed": True, "runtime": runtime}


def test_summary_accounts_for_degraded_rollouts_without_promoting_quality():
    config = binding.read_config()[1]
    branches = [
        simulated_branch(2, "strict"),
        simulated_branch(6, runner.DEGRADED),
        simulated_branch(10, None),
        simulated_branch(11, None),
    ]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["strict_selected"] == counts["degraded_selected"] == 1
    assert counts["shifted_rollouts"] == 2 and counts["quality_response_evaluations"] == 9
    branches[1]["runtime"]["model_safe_tracking_fallback_traces"][0]["response_evaluations"] = 321
    with pytest.raises(ValueError, match="response budget"):
        runner.summarize_phase(branches, "producer", config)


def test_checker_starts_without_candidate_modules_in_a_fresh_process():
    code = "from scripts.simulation.runtime import run_model_safe_tracking_fallback as r; r.require_checker_independence(); print('independent')"
    process = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30)
    assert process.stdout.strip() == "independent"
