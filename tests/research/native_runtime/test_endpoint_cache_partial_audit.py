"""Audit a completed replay without rewriting its failed owner receipt."""

import copy

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_endpoint_cache_replay as runner
from scripts.verification.reference_ports import audit_endpoint_cache_partial as audit


def test_cached_replay_is_audited_with_original_owner_still_failed():
    result = audit.load_audit()
    assert result["complete"] and result["passed"] and not result["original_owner_passed"]
    assert result["replayed_actions"] == 40 and result["task_success_tail_periods"] == 33
    assert result["replay_max_difference"]["state"] <= 1e-10
    assert result["historical_final_state_sha256"] != result["replay_final_state_sha256"]
    assert result["new_model_requests"] == result["new_physics_steps"] == 0


@pytest.mark.parametrize("change", ["action", "reused_identity", "capture_state"])
def test_tampered_cached_evidence_is_rejected(change):
    inputs, owner, branch = runner.load_inputs(), prefix.read(audit.OWNER), copy.deepcopy(prefix.read(audit.BRANCH))
    if change == "action":
        branch["runtime"]["continuation"]["segments"][0]["continuation"]["periods"][0]["command"][0] += 0.1
    elif change == "reused_identity":
        next(row for row in branch["endpoint_cache_events"] if row["kind"] == "reused")["state_sha256"] = "bad"
    else:
        branch["capture"]["render_after"]["sim_state_sha256"] = "bad"
    with pytest.raises((ValueError, AssertionError)):
        audit.summarize(inputs, owner, branch)
