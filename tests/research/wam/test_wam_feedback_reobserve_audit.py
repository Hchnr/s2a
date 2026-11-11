import copy
import json

import pytest

from scripts.verification.reference_ports import audit_wam_feedback_reobserve as audit


def test_saved_run_has_exact_first_reference_and_released_gpu():
    config = json.loads(audit.CONFIG.read_text())
    audit.validate_config(config)
    result = audit.load_audit(config)
    assert result["causal_first_reference_exact"] is True
    assert [row["step"] for row in result["rows"]] == [80, 88]
    assert result["denoise_calls"] == 70 and len(result["mask_evidence"]) == 2
    assert result["gpu_worker_exit"] == 0 and result["gpu_released"] is True
    assert result["previous_attempt_failed"] is result["earlier_partial_model_failed"] is True
    assert result["new_model_requests"] == result["new_physics_steps"] == 0


def test_audit_rejects_missing_or_misordered_denoise_calls():
    config = json.loads(audit.CONFIG.read_text())
    model = audit.read_ref(config, "model_result")
    audit.validate_ledger(model["denoise_evidence"], model["mask_evidence"])
    ledger = copy.deepcopy(model["denoise_evidence"])
    ledger[35]["label"] = ledger[34]["label"]
    with pytest.raises(ValueError, match="request order"):
        audit.validate_ledger(ledger, model["mask_evidence"])
    masks = copy.deepcopy(model["mask_evidence"])
    masks[1]["inactive_rms_after"] = 0.1
    with pytest.raises(ValueError, match="per completed request"):
        audit.validate_ledger(model["denoise_evidence"], masks)
