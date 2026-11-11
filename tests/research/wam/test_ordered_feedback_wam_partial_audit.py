import copy
import json

import numpy as np
import pytest

from s2a.references.action_reference import encode_oracle_targets
from scripts.wam.tracking.preview import audit_ordered_feedback_wam_partial_20261111 as audit


def test_saved_responses_keep_original_failure_and_missing_ledger_explicit():
    config = json.loads(audit.CONFIG.read_text())
    audit.validate_config(config)
    result = audit.load_audit(config)
    assert result["original_worker_success"] is False
    assert result["mask_initial_noise_ledger_available"] is False
    assert result["producer_self_reported_denoise_completed"] == 70
    assert [row["step"] for row in result["rows"]] == [80, 88]
    assert result["new_model_requests"] == result["new_physics_steps"] == 0


def test_audit_rejects_privileged_request_and_nonzero_inactive_output():
    config = json.loads(audit.CONFIG.read_text())
    original = json.loads(audit.ORIGINAL_CONFIG.read_text())
    image, expected = original["images"][0], config["responses"][0]
    directory = audit.INFERENCE / "model_run/step80"
    request = audit.repro.read_checked(directory / "request.json", expected["request_sha256"])
    response = audit.repro.read_checked(directory / "response.json", expected["response_sha256"])
    conditioning = audit.repro.read_checked(directory / "conditioning.json", expected["conditioning_sha256"])
    runtime = audit.read_ref(config, "runtime")
    assert audit.validate_request_response(request, response, conditioning, image, runtime).shape == (40, 128)
    altered = copy.deepcopy(request)
    altered["future_state"] = [0]
    with pytest.raises(ValueError, match="conditioning"):
        audit.validate_request_response(altered, response, conditioning, image, runtime)
    altered_response = copy.deepcopy(response)
    altered_response["outputs"][0]["content"]["action"][0][10] = 1
    with pytest.raises(ValueError, match="inactive slots"):
        audit.validate_request_response(request, altered_response, conditioning, image, runtime)


def test_saved_response_score_integrates_from_current_tcp():
    anchor = np.eye(4)
    anchor[0, 3] = 0.2
    targets = np.repeat(anchor[None], 40, axis=0)
    targets[:, 0, 3] += np.arange(1, 41) * 0.001
    openings = np.full(40, 0.5)
    actions = encode_oracle_targets(anchor, targets, openings)
    actions[0, 0] += 0.01
    actions[:, 9] += 0.1
    score = audit.score_saved_response(anchor, actions, targets, openings)
    for horizon in ("h8", "h40"):
        assert score[horizon]["position_rmse_m"] == pytest.approx(0.01)
        assert score[horizon]["rotation_rmse_rad"] == pytest.approx(0, abs=1e-12)
        assert score[horizon]["opening_rmse"] == pytest.approx(0.1)
