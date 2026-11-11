"""Two same-step current images must retain their own branch identity."""

import copy
from pathlib import Path

import pytest

from scripts.wam.tracking.preview import early_tabletop_live_wam as live


def test_same_step_requests_are_sent_to_separate_branch_directories(tmp_path):
    images = [{"mode": "original", "step": 120}, {"mode": "pose_bounded", "step": 120}]
    seen, responses = [], []

    def delegate(config, _ctx, _service, _torch, _counter, _overrides, _domain, directory, _template, result, output):
        seen.append((config["images"][0]["mode"], Path(directory).name))
        output.append({"step": config["images"][0]["step"]})
        result["count"] += 1

    result = {"count": 0}
    live.split_branch_requests(
        {"images": images}, None, None, None, None, None, None, tmp_path, None, result, responses, delegate=delegate
    )
    assert seen == [("original", "original"), ("pose_bounded", "pose_bounded")]
    assert [row["mode"] for row in responses] == ["original", "pose_bounded"] and result["count"] == 2


def test_image_cannot_be_bound_to_the_other_branch_state():
    image = {
        "mode": "original",
        "step": 120,
        "path": "outputs/example/current.png",
        "sha256": "a" * 64,
        "state_sha256": "b" * 64,
    }
    capture = {
        "frame_index": 120,
        "canvas_png": {"path": str(live.ROOT / image["path"]), "sha256": image["sha256"]},
        "current_endpoint": {"sim_state_sha256": image["state_sha256"], "sim_time": 6.0},
        "render_before": {"sim_state_sha256": image["state_sha256"], "sim_time": 6.0},
        "render_after": {"sim_state_sha256": image["state_sha256"], "sim_time": 6.0},
        "future_rgb_frames": 0,
        "camera_roles": ["auxiliary_0", "left_wrist"],
        "canvas_shape": [960, 640, 3],
        "canvas_dtype": "uint8",
    }
    live.validate_binding(image, capture)
    swapped = copy.deepcopy(capture)
    swapped["current_endpoint"]["sim_state_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        live.validate_binding(image, swapped)
