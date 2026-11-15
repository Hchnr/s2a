"""Validate that saved references are bound to the actual observed state."""

import numpy as np
import pytest

from scripts.simulation.runtime import bounded_reference_choice as comparison
from scripts.simulation.runtime import run_bounded_reference_choice as runner


def test_saved_reference_rejects_mismatched_image_or_state(monkeypatch):
    actions = np.zeros((40, 128))
    actions[:, 3], actions[:, 7], actions[:, 9] = 1.0, 1.0, 1.0
    image_sha, state_sha = "image", "state"
    capture = {
        "frame_index": 8,
        "current_endpoint": {"sim_time": 0.4, "sim_state_sha256": state_sha},
        "canvas_png": {"sha256": image_sha},
        "current_source_pose": np.eye(4),
    }
    request = {"step": 8, "sim_time_s": 0.4, "state_sha256": state_sha, "image": {"sha256": image_sha}, "seed": 43}
    envelope = {
        "step": 8,
        "sim_time_s": 0.4,
        "state_sha256": state_sha,
        "image_sha256": image_sha,
        "seed": 43,
        "denoise_calls": 35,
        "action": actions,
        "normalized_action": actions,
    }
    raw = {
        "status": "success",
        "outputs": [{"content": {"action": actions, "action128": {"normalized_action": actions}}}],
    }
    lookup = {"request": request, "envelope": envelope, "raw": raw}
    monkeypatch.setattr(comparison.prefix, "read", lambda ref: lookup[ref])
    exchange = {
        "step": 8,
        "request_ref": "request",
        "response_ref": "envelope",
        "request_sent": True,
        "response_received": True,
        "response_accepted": True,
    }
    response = {"step": 8, "actual_response_ref": "raw"}
    poses, opening = comparison.saved_reference(capture, exchange, response, np.eye(4), 8)
    np.testing.assert_allclose(poses, np.tile(np.eye(4), (40, 1, 1)))
    np.testing.assert_allclose(opening, np.ones(40))
    envelope["image_sha256"] = "other image"
    with pytest.raises(ValueError, match="original current image"):
        comparison.saved_reference(capture, exchange, response, np.eye(4), 8)
    envelope["image_sha256"] = image_sha
    request["state_sha256"] = "future state"
    with pytest.raises(ValueError, match="original current image"):
        comparison.saved_reference(capture, exchange, response, np.eye(4), 8)


def test_interrupted_progress_keeps_attempted_and_completed_separate():
    receipt = {
        "control_steps_attempted": 4,
        "control_steps_completed": 3,
        "counters": {"mj_step": {"attempted": 97, "completed": 96}},
    }
    event = runner.progress_event("producer", "old", receipt)
    assert event["control_steps_attempted"] == 4
    assert event["control_steps_completed"] == 3
    assert event["native_steps"] == {"attempted": 97, "completed": 96}
    assert event["models"] is None and event["contexts"] is None
    event["native_steps"]["completed"] = 0
    assert receipt["counters"]["mj_step"]["completed"] == 96
    with pytest.raises(ValueError, match="registered phase"):
        runner.progress_event("producer", "unknown", receipt)
