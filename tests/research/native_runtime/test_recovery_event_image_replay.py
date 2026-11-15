"""New event images must observe an exact endpoint during saved-command replay."""

from pathlib import Path

import pytest

from s2a.research.native_runtime.development_prefix_replay import ReplayObservers
from s2a.research.native_runtime.recovery_event_image_replay import event_image_replay


def test_event_images_bind_actual_endpoint_and_keep_legacy_renders_unsaved(tmp_path):
    actual_receipts = [dict(endpoint_step=i, sim_state_sha256=str(i), sim_time=i * 0.05) for i in range(5)]
    saved_images, camera_calls = [], []

    def endpoint(_env, step):
        return dict(endpoint_step=step), actual_receipts[step].copy()

    def render(_env, step, _sample, receipt, directory):
        assert Path(directory).is_relative_to(tmp_path)
        saved_images.append(step)
        binding = dict(sim_time=receipt["sim_time"], sim_state_sha256=receipt["sim_state_sha256"])
        return dict(frame_index=step, current_endpoint=receipt, render_before=binding, render_after=binding)

    def replay(env, commands, _saved, _receipts, output, observers, receipt, **_kwargs):
        for step in range(len(commands) + 1):
            sample, current = observers.endpoint(env, step)
            if step == 0:
                observers.render_anchor(env, step, sample, current, output)
                observers.render_cameras(env, 256)
        return receipt

    observers = ReplayObservers(endpoint, lambda *_args: None, render, lambda *_args: camera_calls.append(True))
    receipt = {}
    event_image_replay(replay, [2, 4])(None, [0] * 4, {}, actual_receipts, tmp_path, observers, receipt)
    assert saved_images == [2, 4] and camera_calls == [True]
    assert [row["frame_index"] for row in receipt["event_images"]] == [2, 4]
    assert receipt["unsaved_current_camera_renders"] == 1 and "event_capture_partial" not in receipt


def test_event_replay_rejects_changed_actual_receipt_before_render(tmp_path):
    actual = [dict(endpoint_step=i, sim_state_sha256=str(i)) for i in range(3)]
    expected = [dict(item) for item in actual]
    expected[1]["sim_state_sha256"] = "changed"
    images = []
    observers = ReplayObservers(
        lambda _env, step: ({}, actual[step]),
        lambda *_args: None,
        lambda *_args: images.append(True),
        lambda *_args: None,
    )

    def replay(env, _commands, _saved, _receipts, _output, wrapped, _receipt, **_kwargs):
        wrapped.endpoint(env, 1)

    receipt = {}
    with pytest.raises(ValueError, match="original actual receipt"):
        event_image_replay(replay, [1])(None, [0, 0], {}, expected, tmp_path, observers, receipt)
    assert images == [] and receipt["event_capture_partial"] == []
    with pytest.raises(ValueError, match="Ordered, distinct"):
        event_image_replay(replay, [2, 2])
