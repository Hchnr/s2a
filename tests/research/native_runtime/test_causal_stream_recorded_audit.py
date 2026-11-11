"""Audit actual CPU artifact identities and independent physical replay."""

import copy
import json
from pathlib import Path

import pytest

from scripts.simulation.runtime import causal_stream_recorded_admission as admission
from scripts.verification.reference_ports import audit_causal_stream_recorded_physical as audit


def exchange_inputs():
    inputs = admission.load_inputs()
    summary = audit.prefix.read({"path": str(audit.PRODUCER), "sha256": audit.PRODUCER_SHA})
    branch = audit.prefix.read(summary["branch_ref"])
    runtime = branch["runtime"]
    return inputs, runtime["streaming_captures"][0], runtime["streaming_model_exchanges"][0]


def test_actual_physical_stream_and_checker_preserve_evidence_boundaries():
    result = audit.load_audit()
    assert result["complete"] and result["passed"] and not result["control_qualified"]
    assert result["new_model_requests"] == result["new_denoise_calls"] == result["new_audit_physics_steps"] == 0
    assert result["recorded_producer_native_steps"] == result["recorded_checker_native_steps"] == 2400
    assert result["replayed_prefix_inner_samples"] == 2000
    assert result["replayed_continuation_inner_samples"] == 400
    assert result["max_replay_joint_difference_rad"] == result["max_replay_velocity_difference"] == 0
    assert result["max_replay_time_difference_s"] == 0
    assert [row["step"] for row in result["rows"]] == [80, 88]
    first, second = result["windows"]
    assert first["position_rmse_m"] == pytest.approx(0.002405181, abs=1e-9)
    assert second["position_rmse_m"] == pytest.approx(0.002708882, abs=1e-9)
    assert first["pose_caps_satisfied"] == second["pose_caps_satisfied"] == 8
    assert result["terminal_consecutive_success_periods"] == 9
    assert result["qualified_stable_success"] is False


def test_changed_capture_or_incomplete_exchange_is_rejected():
    inputs, capture, exchange = exchange_inputs()
    root = audit.PRODUCER.parent / "simulation/runtime"
    changed = copy.deepcopy(capture)
    changed["render_after"]["sim_time"] += 0.002
    with pytest.raises(ValueError, match="captured current image"):
        audit.verify_exchange(inputs["records"][0], changed, exchange, root, {}, index=0)
    changed = copy.deepcopy(exchange)
    changed["response_accepted"] = False
    with pytest.raises(ValueError, match="completed response exchange"):
        audit.verify_exchange(inputs["records"][0], capture, changed, root, {}, index=0)


def test_changed_saved_response_bytes_fail_the_recorded_sha(tmp_path):
    inputs, capture, exchange = exchange_inputs()
    changed = copy.deepcopy(exchange)
    source = Path(exchange["response_ref"]["path"])
    target = tmp_path / "changed-response.json"
    target.write_bytes(source.read_bytes() + b" ")
    changed["response_ref"]["path"] = str(target)
    root = audit.PRODUCER.parent / "simulation/runtime"
    with pytest.raises(ValueError, match="Changed input"):
        audit.verify_exchange(inputs["records"][0], capture, changed, root, {}, index=0)


def test_forged_request_identity_is_rejected_even_if_its_file_sha_is_updated(tmp_path):
    inputs, capture, exchange = exchange_inputs()
    changed = copy.deepcopy(exchange)
    request = json.loads(Path(exchange["request_ref"]["path"]).read_text())
    request["step"] = 88
    target = tmp_path / "forged-request.json"
    target.write_text(json.dumps(request))
    changed["request_ref"] = audit.prefix.reference(target)
    root = audit.PRODUCER.parent / "simulation/runtime"
    with pytest.raises(ValueError, match="stored request and response identity"):
        audit.verify_exchange(inputs["records"][0], capture, changed, root, {}, index=0)
