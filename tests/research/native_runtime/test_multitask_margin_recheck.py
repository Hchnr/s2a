"""The multi-task probe binds its fixed roster and each generated reference."""

import copy
import json

import numpy as np
import pytest

from s2a.geometry import matrix_to_rot6d
from scripts.simulation.runtime import run_multitask_margin_recheck as runner


def test_registered_roster_and_budgets_are_not_mutable():
    config = json.loads(runner.CONFIG.read_text())
    runner.validate_config(config)
    for key, value in (
        ("max_extra_periods_per_case", 2),
        ("max_total_control_steps_per_phase", 600),
        ("automatic_retry", 1),
    ):
        altered = dict(config, **{key: value})
        with pytest.raises(ValueError):
            runner.validate_config(altered)
    altered = copy.deepcopy(config)
    altered["cases"][0]["trigger_step"] += 1
    with pytest.raises(ValueError):
        runner.validate_config(altered)


def test_saved_reference_requires_matching_state_image_seed_and_actual_response(monkeypatch):
    step, seed = 16, 42
    actions = np.zeros((40, 128))
    actions[:, 3:9] = matrix_to_rot6d(np.eye(3))
    actions[:, 9] = 0.8
    raw = dict(
        status="success", outputs=[dict(content=dict(action=actions, action128=dict(normalized_action=actions)))]
    )
    request = dict(step=step, seed=seed, sim_time_s=0.8, state_sha256="state", image=dict(sha256="image"))
    envelope = dict(
        step=step,
        seed=seed,
        sim_time_s=0.8,
        state_sha256="state",
        image_sha256="image",
        denoise_calls=35,
        action=actions,
        normalized_action=actions,
    )
    capture = dict(
        frame_index=step,
        current_endpoint=dict(sim_time=0.8, sim_state_sha256="state"),
        canvas_png=dict(sha256="image"),
        current_source_pose=np.eye(4),
    )
    exchange = dict(
        step=step,
        request_ref="request",
        response_ref="envelope",
        request_sent=True,
        response_received=True,
        response_accepted=True,
    )
    response = dict(step=step, reference_sent=True, actual_response_ref="raw")
    records = dict(request=request, envelope=envelope, raw=raw)
    monkeypatch.setattr(runner.prefix, "read", lambda ref: records[ref])
    poses, openings = runner.saved_reference(capture, exchange, response, np.eye(4), step=step, seed=seed)
    np.testing.assert_allclose(poses, np.tile(np.eye(4), (40, 1, 1)), atol=1e-12)
    np.testing.assert_allclose(openings, 0.8)
    with pytest.raises(ValueError, match="current image"):
        runner.saved_reference(capture, exchange, response, np.eye(4), step=step, seed=43)
    request["state_sha256"] = "different"
    with pytest.raises(ValueError, match="current image"):
        runner.saved_reference(capture, exchange, response, np.eye(4), step=step, seed=seed)
