"""A WAM image must belong to the unchanged branch endpoint."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime.early_tabletop_continuation import capture_current_image


def example(tmp_path, *, advance_during_render=False):
    state = np.array([6.0, 0.1, -0.2])
    sim = SimpleNamespace(get_state=lambda: SimpleNamespace(flatten=lambda: state.copy()))
    env = SimpleNamespace(sim=sim)
    endpoint = {"endpoint_step": 120, "sim_time": 6.0, "sim_state_sha256": state_sha256(state)}
    image = tmp_path / "current.png"
    image.write_bytes(b"example-current-image")
    image_ref = {"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}

    def render(_env, step, _sample, receipt, _directory):
        if advance_during_render:
            state[1] += 0.01
        binding = {"sim_time": 6.0, "sim_state_sha256": endpoint["sim_state_sha256"]}
        return {
            "frame_index": step,
            "current_endpoint": receipt,
            "render_before": binding,
            "render_after": binding.copy(),
            "future_rgb_frames": 0,
            "camera_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "canvas_shape": [960, 640, 3],
            "canvas_dtype": "uint8",
            "canvas_png": image_ref,
        }

    observers = SimpleNamespace(endpoint=lambda _env, _step: ({}, endpoint.copy()), render_anchor=render)
    return env, observers


def test_current_image_binds_exact_completed_state(tmp_path):
    env, observers = example(tmp_path)
    result = capture_current_image(env, observers, step=120, directory=tmp_path)
    assert result["capture"]["future_rgb_frames"] == 0
    assert result["capture"]["current_endpoint"]["sim_state_sha256"] == result["endpoint"]["sim_state_sha256"]


def test_image_render_cannot_advance_or_replace_state(tmp_path):
    env, observers = example(tmp_path, advance_during_render=True)
    with pytest.raises(ValueError):
        capture_current_image(env, observers, step=120, directory=tmp_path)
