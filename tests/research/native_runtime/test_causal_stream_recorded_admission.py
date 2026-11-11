"""Freeze the previously audited references and CPU-only reproduction budget."""

import copy
import json

import pytest

from scripts.simulation.runtime import causal_stream_recorded_admission as admission


def test_real_saved_inputs_and_independent_historical_replay_are_bound():
    inputs = admission.load_inputs()
    assert [row["step"] for row in inputs["records"]] == [80, 88]
    assert [row["step"] for row in inputs["periods"]] == list(range(80, 96))
    assert inputs["model_sha256"] == admission.FIXED["historical_weight_manifest_sha256"]
    assert inputs["producer"]["runtime"]["counters"]["mj_step"]["completed"] == 2400
    assert inputs["checker"]["runtime"]["counters"]["mj_step"]["completed"] == 2400
    assert len(inputs["source_sha256"]) >= 150
    json.dumps(admission.input_receipt(inputs), allow_nan=False)


@pytest.mark.parametrize(
    "key", ["windows", "max_native_steps_per_phase", "new_model_requests", "seed", "automatic_retry"]
)
def test_unregistered_budget_or_seed_is_rejected(key):
    config = dict(admission.FIXED)
    config[key] = True if key != "automatic_retry" else 1
    with pytest.raises(ValueError, match="budget must remain fixed"):
        admission.validate_config(config)


def test_exact_response_gate_rejects_changed_current_image_or_reference():
    record = admission.load_inputs()["records"][0]
    request = {
        "request_id": 0,
        "step": record["step"],
        "sim_time_s": record["sim_time_s"],
        "state_sha256": record["state_sha256"],
        "image": {"sha256": record["image_sha256"]},
    }
    response = {
        "image_sha256": record["image_sha256"],
        "action": copy.deepcopy(record["action"]),
        "normalized_action": copy.deepcopy(record["normalized_action"]),
    }
    assert admission.gate_response([record], request, response) is True
    wrong = copy.deepcopy(request)
    wrong["image"]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="observation differs"):
        admission.gate_response([record], wrong, response)
    response["normalized_action"][0][0] += 0.01
    with pytest.raises(ValueError, match="normalized_action changed"):
        admission.gate_response([record], request, response)


def test_completed_period_guard_checks_the_actual_recorded_inner_samples():
    periods = admission.load_inputs()["periods"]
    assert admission.guard_period(periods, copy.deepcopy(periods[0])) is True
    changed = copy.deepcopy(periods[0])
    changed["actual_inner"][0]["q"][0] += 0.01
    with pytest.raises(AssertionError):
        admission.guard_period(periods, changed)


def test_checkpoint_manifest_identity_detects_a_changed_historical_digest():
    manifest = admission.prefix.read(admission._configured_ref(admission.FIXED, "historical_checkpoint_manifest"))
    changed = copy.deepcopy(manifest)
    entry = next(row for row in changed["checkpoints"] if row["name"] == "8400_masked")
    entry["files"][next(iter(entry["files"]))]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="weight manifest changed"):
        admission._fingerprint(changed, admission.FIXED["historical_weight_manifest_sha256"])
