"""Audit the actual bounded live model run without starting another model."""

import copy
import json
from pathlib import Path

import pytest

from scripts.verification.reference_ports import audit_causal_stream_live_wam as audit
from scripts.wam.tracking.preview import causal_stream_live_admission as admission


def inputs_and_first_response():
    inputs = admission.load_inputs()
    summary = audit.prefix.read({"path": str(audit.RUN / "producer/result.json"), "sha256": audit.PRODUCER_SHA})
    model = audit.prefix.read(summary["model_ref"])
    producer = audit.prefix.read(summary["branch_ref"])
    return inputs, model, producer["runtime"]["streaming_model_exchanges"][0]


def test_live_wam_and_independent_physics_are_audited_without_new_work():
    result = audit.load_audit()
    assert result["complete"] and result["passed"] and not result["control_qualified"]
    assert result["model_initializations"] == 1 and result["model_requests"] == 2 and result["denoise_calls"] == 70
    assert result["new_audit_model_requests"] == result["new_audit_physics_steps"] == 0
    assert result["replayed_prefix_inner_samples"] == 2000 and result["replayed_continuation_inner_samples"] == 400
    assert result["checker_policy_isolated"]
    assert result["max_replay_joint_difference_rad"] == result["max_replay_velocity_difference"] == 0
    assert result["max_replay_time_difference_s"] == 0
    assert result["actually_imported_cosmos_source_files"] > 100
    assert [row["step"] for row in result["rows"]] == [80, 88]
    assert all(row["model_elapsed_seconds"] > 0.4 for row in result["rows"])
    assert result["terminal_consecutive_success_periods"] == 9
    assert result["qualified_stable_success"] is False


def test_incomplete_denoise_ledger_is_rejected():
    _inputs, model, _exchange = inputs_and_first_response()
    ledger = copy.deepcopy(model["denoise_evidence"])
    ledger[10]["completed"] = False
    with pytest.raises(ValueError, match="denoise call must complete"):
        audit.validate_ledger(ledger, model["mask_evidence"])


def test_changed_delivered_reference_is_detected_independently(tmp_path):
    inputs, model, exchange = inputs_and_first_response()
    changed = copy.deepcopy(exchange)
    envelope = json.loads(Path(exchange["response_ref"]["path"]).read_text())
    envelope["action"][0][0] += 0.01
    target = tmp_path / "changed-envelope.json"
    target.write_text(json.dumps(envelope))
    changed["response_ref"] = audit.prefix.reference(target)
    with pytest.raises(ValueError, match="Delivered reference differs"):
        audit.verify_model_response(inputs, inputs["records"][0], model["responses"][0], changed, {})


def test_changed_actual_wam_output_is_rejected_even_with_an_updated_file_sha(tmp_path):
    inputs, model, exchange = inputs_and_first_response()
    row = copy.deepcopy(model["responses"][0])
    response = json.loads(Path(row["actual_response_ref"]["path"]).read_text())
    response["outputs"][0]["content"]["action"][0][0] += 0.01
    target = tmp_path / "changed-model-response.json"
    target.write_text(json.dumps(response))
    row["actual_response_ref"] = audit.prefix.reference(target)
    with pytest.raises(ValueError, match="differs from the audited reference"):
        audit.verify_model_response(inputs, inputs["records"][0], row, exchange, {})
