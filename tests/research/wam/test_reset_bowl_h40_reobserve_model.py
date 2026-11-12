"""Bind the offline model request to the verified current observation."""

import copy

import pytest

from scripts.simulation.runtime import run_reset_bowl_h40_reobserve_model as owner


def valid_capture():
    binding = {"sim_time": 2.65, "sim_state_sha256": owner.STATE_SHA}
    return dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        control_qualified=False,
        counts=dict(control_steps=48, native_steps=1200, generated_reference_steps=40, stop=None),
        independent_max_absolute_difference={"command": 0.0, "state": 0.0},
        capture=dict(
            frame_index=48,
            current_endpoint={"endpoint_step": 48, **binding},
            render_before=dict(binding),
            render_after=dict(binding),
            future_rgb_frames=0,
            canvas_png={"path": "unused.png", "sha256": owner.IMAGE_SHA},
            camera_roles=["auxiliary_0", "left_wrist"],
            canvas_shape=[960, 640, 3],
        ),
    )


def test_accepts_only_the_fully_replayed_endpoint():
    assert owner.validate_capture(valid_capture()) == dict(step=48, sim_time_s=2.65, state_sha256=owner.STATE_SHA)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("counts", "control_steps", 47),
        ("counts", "stop", {"reason": "early"}),
        ("independent_max_absolute_difference", "state", 1e-5),
        ("capture", "frame_index", 40),
        ("capture", "future_rgb_frames", 1),
        ("capture", "camera_roles", ["left_wrist"]),
    ],
)
def test_rejects_partial_or_mismatched_capture(section, key, value):
    result = valid_capture()
    result[section][key] = value
    with pytest.raises(ValueError, match="step-48 image and state"):
        owner.validate_capture(result)


def test_rejects_a_changed_image_or_rendered_state():
    for section, key in (("canvas_png", "sha256"), ("render_after", "sim_state_sha256")):
        result = valid_capture()
        result["capture"][section][key] = "a" * 64
        with pytest.raises(ValueError, match="step-48 image and state"):
            owner.validate_capture(result)


def test_model_worker_selects_the_bowl_task_and_registered_device(monkeypatch):
    from scripts.wam.tracking.preview import reset_live_request

    calls = []

    def fake_serve(**kwargs):
        return kwargs["request_builder"]({"template": True}, {"step": 48})

    def fake_request(template, image, task):
        calls.append((copy.deepcopy(template), copy.deepcopy(image), task))
        return "reference"

    monkeypatch.setattr(reset_live_request, "serve_one", fake_serve)
    monkeypatch.setattr(owner.tasks, "make_task_request", fake_request)
    monkeypatch.setattr(owner.live, "OUTPUT", None)
    monkeypatch.setattr(owner.live, "GPU", None)
    monkeypatch.setattr(owner.live, "model_worker_inputs", None)
    monkeypatch.setattr(owner.live, "model_worker", lambda: reset_live_request.serve_one())
    owner.model_worker()
    assert calls == [({"template": True}, {"step": 48}, owner.TASK)]
    assert owner.live.OUTPUT == owner.OUTPUT and owner.live.GPU == owner.GPU
