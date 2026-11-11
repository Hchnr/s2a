"""Independent camera replay must match the branch state and image bytes."""

import copy
import hashlib

import numpy as np
import pytest

from scripts.simulation.runtime.run_early_tabletop_capture import compare_capture


def example(tmp_path):
    path = tmp_path / "current.png"
    path.write_bytes(b"current-image")
    ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    capture = {
        "frame_index": 120,
        "canvas_png": ref,
        "canvas_rgb_sha256": "a" * 64,
        "future_rgb_frames": 0,
        "current_source_pose": np.eye(4),
        "current_opening": 0.5,
    }
    image = {"endpoint": {"sim_state_sha256": "b" * 64, "sim_time": 6.0}, "capture": capture}
    return {"runtime": {"continuation": {"next_current_image": image}}}


def test_captured_image_and_endpoint_must_match(tmp_path):
    producer = example(tmp_path)
    replay = copy.deepcopy(producer)
    assert (
        compare_capture(producer, replay)["image_ref"]["sha256"]
        == producer["runtime"]["continuation"]["next_current_image"]["capture"]["canvas_png"]["sha256"]
    )
    replay["runtime"]["continuation"]["next_current_image"]["capture"]["canvas_rgb_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        compare_capture(producer, replay)
