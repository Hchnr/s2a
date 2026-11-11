"""Independent replay must bind the same actual image and state."""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime.reset_live_two_window_replay import compare_capture


def capture():
    return dict(
        frame_index=8,
        current_endpoint={"endpoint_step": 8, "sim_time": 0.65, "sim_state_sha256": "a" * 64},
        render_before={"sim_time": 0.65, "sim_state_sha256": "a" * 64},
        render_after={"sim_time": 0.65, "sim_state_sha256": "a" * 64},
        future_rgb_frames=0,
        camera_keys=["agentview_image", "robot0_eye_in_hand_image"],
        canvas_shape=[960, 640, 3],
        canvas_dtype="uint8",
        canvas_rgb_sha256="b" * 64,
        canvas_png={"path": "/old/current.png", "sha256": "c" * 64},
        current_source_pose=np.eye(4).tolist(),
        current_opening=0.7,
    )


def test_capture_comparison_rejects_different_pixels_or_state_but_allows_own_path():
    expected, actual = capture(), capture()
    actual["canvas_png"]["path"] = "/new/current.png"
    compare_capture(expected, actual)
    for key in ("canvas_rgb_sha256", "render_after", "current_source_pose"):
        changed = copy.deepcopy(actual)
        if key == "render_after":
            changed[key]["sim_time"] += 0.002
        elif key == "current_source_pose":
            changed[key][0][3] = 0.1
        else:
            changed[key] = "d" * 64
        with pytest.raises((ValueError, AssertionError)):
            compare_capture(expected, changed)


def test_checker_import_does_not_load_candidate_or_simulator():
    code = "from scripts.simulation.runtime import reset_live_two_window_replay as s; s.isolation.checker_isolation(); s.original.require_no_simulator_import()"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
