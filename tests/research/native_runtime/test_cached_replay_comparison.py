"""Bind capture to its actual replay, keeping historical differences visible."""

import copy

import pytest

from s2a.research.native_runtime.cached_replay_comparison import compare_cached_replay
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_endpoint_cache_replay as runner
from scripts.verification.reference_ports import audit_endpoint_cache_fixed_comparison as fixed
from scripts.verification.reference_ports import audit_endpoint_cache_partial as audit


def recorded():
    inputs = runner.load_inputs()
    branch = prefix.read(audit.BRANCH)
    return inputs["base"]["stream"], branch["runtime"], branch["endpoint_cache_events"], branch["capture"]


def test_actual_capture_binds_replay_even_when_historical_state_bytes_differ():
    expected, runtime, events, capture = recorded()
    result = compare_cached_replay(runtime, expected, events, capture)
    assert result["completed_actions"] == 40 and not result["final_state_bytes_equal"]
    assert result["max_difference"]["state"] == pytest.approx(6.601403451655941e-13)


def test_current_capture_cannot_be_bound_to_the_historical_digest():
    expected, runtime, events, capture = recorded()
    capture = copy.deepcopy(capture)
    capture["current_endpoint"]["sim_state_sha256"] = expected["segments"][-1]["continuation"]["periods"][-1][
        "endpoint_receipt"
    ]["sim_state_sha256"]
    with pytest.raises(ValueError, match="actual replay endpoint"):
        compare_cached_replay(runtime, expected, events, capture)


def test_changed_inner_replay_beyond_tolerance_is_rejected():
    expected, runtime, events, capture = recorded()
    runtime = copy.deepcopy(runtime)
    runtime["continuation"]["segments"][0]["continuation"]["periods"][0]["actual_inner"][0]["q"][0] += 1e-4
    with pytest.raises(ValueError, match="cross-run physical tolerance"):
        compare_cached_replay(runtime, expected, events, capture)


def test_corrected_scorer_agrees_with_independent_audit_without_new_work():
    result = fixed.load_audit()
    assert result["complete"] and result["passed"] and not result["original_owner_passed"]
    assert result["comparison"]["completed_actions"] == 40
    assert result["new_model_requests"] == result["new_physics_steps"] == 0
