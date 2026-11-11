import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from s2a.references.action_reference import encode_oracle_targets
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as inference


@pytest.fixture(scope="module")
def registered():
    config = json.loads(inference.CONFIG.read_text())
    inference.validate_config(config)
    captures = inference.capture_inputs(config)
    _prepared, _bridge, _completed, template, _response = inference.previous_inputs()
    return config, captures, template


def test_only_two_current_images_from_reaped_capture_are_admitted(registered):
    config, captures, template = registered
    assert [row["frame_index"] for row in captures] == [80, 88]
    assert all(row["render_before"] == row["render_after"] for row in captures)
    assert all(inference.make_request(template, image)["seed"] == 42 for image in config["images"])


@pytest.mark.parametrize("change", ["future", "mask", "seed", "image", "video", "roles", "task"])
def test_request_rejects_other_conditioning_and_privileged_inputs(registered, change):
    config, _captures, template = registered
    image = config["images"][0]
    request = inference.make_request(template, image)
    if change == "future":
        request["future_state"] = [1]
    elif change == "mask":
        request["extra"]["action128"]["action_dim_mask"][10] = True
    elif change == "seed":
        request["seed"] = 43
    elif change == "image":
        request["vision_path"] += ".other"
    elif change == "video":
        request["extra"]["action128"]["return_video"] = True
    elif change == "roles":
        request["extra"]["action128"]["selected_roles"].reverse()
    else:
        request["prompt"] = "other task"
    with pytest.raises(ValueError):
        inference.validate_request(request, image)


def test_budget_rejects_more_requests_or_changed_gpu(registered):
    config = copy.deepcopy(registered[0])
    config["model_requests"] = 3
    with pytest.raises(ValueError, match="budget"):
        inference.validate_config(config)
    config["model_requests"] = 2
    config["gpu_uuid"] = "GPU-other"
    with pytest.raises(ValueError, match="budget"):
        inference.validate_config(config)


def test_qa_source_binding_rejects_a_changed_inference_script():
    snapshot = {"sources": {"runner.py": "before", "test.py": "same"}}
    inference.validate_qa_snapshot(snapshot, {"runner.py": "before", "test.py": "same"})
    with pytest.raises(ValueError, match="current inference sources"):
        inference.validate_qa_snapshot(snapshot, {"runner.py": "after", "test.py": "same"})


def test_mask_evidence_tracks_each_of_two_requests():
    row = {"stage": "initial_noise", "inactive_rms_before": 1.0, "inactive_rms_after": 0.0}
    inference.validate_mask_evidence([row, row.copy()], 2)
    with pytest.raises(ValueError, match="per completed request"):
        inference.validate_mask_evidence([row], 2)
    bad = dict(row, inactive_rms_after=0.1)
    with pytest.raises(ValueError, match="per completed request"):
        inference.validate_mask_evidence([row, bad], 2)


def test_inference_context_keeps_outputs_in_the_repository():
    context = inference.default_context()
    assert inference.checked_context(context) is context
    assert context.capture_loader is inference.capture_inputs
    with pytest.raises(ValueError, match="repository-local"):
        inference.checked_context(replace(context, output=Path("/tmp/external-model-output")))


def test_offline_score_uses_integrated_tcp_pose_and_absolute_opening():
    anchor = np.eye(4)
    targets = np.repeat(anchor[None], 40, axis=0)
    targets[:, 0, 3] = np.arange(1, 41) * 0.001
    opening = np.full(40, 0.5)
    actions = encode_oracle_targets(anchor, targets, opening)
    exact = inference.score_prediction(anchor, actions, targets, opening)
    assert exact["h8"]["position_rmse_m"] == pytest.approx(0, abs=1e-12)
    assert exact["h40"]["rotation_rmse_rad"] == pytest.approx(0, abs=1e-12)
    actions[0, 0] += 0.01
    actions[:, 9] += 0.1
    changed = inference.score_prediction(anchor, actions, targets, opening)
    for horizon in ("h8", "h40"):
        assert changed[horizon]["position_rmse_m"] == pytest.approx(0.01)
        assert changed[horizon]["opening_rmse"] == pytest.approx(0.1)
