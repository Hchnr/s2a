"""Audit actual six-request and 45-action evidence without new work."""

import copy
import json
from pathlib import Path

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.reference_ports import audit_causal_stream_live_cached as audit
from scripts.wam.tracking.preview import causal_stream_live_cached_admission as admission


def test_cached_live_run_and_independent_replay_are_audited():
    result = audit.load_audit()
    assert result["complete"] and result["passed"] and not result["control_qualified"]
    assert result["model_initializations"] == 1 and result["model_requests"] == 6 and result["denoise_calls"] == 210
    assert result["new_audit_model_requests"] == result["new_audit_physics_steps"] == 0
    assert result["replayed_prefix_inner_samples"] == 2000 and result["replayed_continuation_inner_samples"] == 1125
    assert result["checker_policy_isolated"] and all(value == 0 for value in result["replay_max_difference"].values())
    assert result["terminal_consecutive_success_periods"] == 38 and not result["qualified_stable_success"]
    assert result["producer_stop"]["at_step"] == 125 and not result["stop_reason_independently_verified"]
    assert [row["periods"] for row in result["windows"]] == [8, 8, 8, 8, 8, 5]


def test_changed_delivered_reference_is_not_confused_with_actual_model_output(tmp_path):
    inputs = admission.load_inputs()
    summary = prefix.read({"path": str(audit.RUN / "producer/result.json"), "sha256": audit.PRODUCER_SHA})
    model = prefix.read(summary["model_ref"])
    producer = prefix.read(summary["branch_ref"])
    capture = producer["runtime"]["streaming_captures"][2]
    exchange = copy.deepcopy(producer["runtime"]["streaming_model_exchanges"][2])
    response = json.loads(Path(exchange["response_ref"]["path"]).read_text())
    response["action"][0][0] += 0.01
    target = tmp_path / "changed.json"
    target.write_text(json.dumps(response))
    exchange["response_ref"] = prefix.reference(target)
    with pytest.raises(ValueError, match="actual model response"):
        audit.verify_response(inputs, capture, exchange, model["responses"][2], index=2, consumed={})


def test_metrics_reject_incomplete_action_sets():
    with pytest.raises(ValueError, match="45 completed"):
        audit.metrics([])
