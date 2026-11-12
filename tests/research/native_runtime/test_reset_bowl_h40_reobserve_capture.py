"""The next observation must come from the completely replayed current state."""

import pytest

from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as replay
from scripts.simulation.runtime.run_reset_bowl_h40_reobserve_capture import capture_after_replay
from tests.research.native_runtime.test_reset_bowl_fixed_h40_replay import setup_replay
from tests.research.native_runtime.test_reset_live_two_window import terminal_at


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    monkeypatch.setattr(replay.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(replay.original, "checker_isolation", lambda: None)


def capture(case, receipt, options):
    return capture_after_replay(
        case.env,
        terminal_at(case, 0),
        case.observers,
        receipt,
        checker=replay.run_checker,
        capture_render=lambda render, _scope, *args, **kwargs: render(*args, **kwargs),
        **options,
    )


def test_captures_only_after_complete_original_reference_replay(tmp_path):
    case, receipt, options = setup_replay(tmp_path)
    result = capture(case, receipt, options)
    observation = receipt["reobserve_step48"]
    assert len(result["replays"]) == 5
    assert case.sim.inner == 1200
    assert observation["snapshot"]["step"] == observation["capture"]["frame_index"] == 48
    assert observation["capture"]["future_rgb_frames"] == 0
    assert observation["capture"]["current_endpoint"]["sim_state_sha256"] == observation["snapshot"]["state_sha256"]


def test_early_stop_cannot_be_presented_as_step48(tmp_path):
    case, receipt, options = setup_replay(tmp_path, stop_at=19)
    with pytest.raises(ValueError, match="complete replay"):
        capture(case, receipt, options)
    assert case.sim.inner == 475
    assert "reobserve_step48" not in receipt


def test_render_cannot_change_the_current_physical_state(tmp_path):
    case, receipt, options = setup_replay(tmp_path)
    original = case.observers.render_anchor

    def changed(env, step, *args):
        image = original(env, step, *args)
        if step == 48:
            env.sim.data.time += 0.002
        return image

    case.observers.render_anchor = changed
    with pytest.raises(ValueError, match="without changing the simulation"):
        capture(case, receipt, options)
    assert "reobserve_step48" not in receipt


def test_rendered_endpoint_must_match_the_replayed_state(tmp_path):
    case, receipt, options = setup_replay(tmp_path)
    original = case.observers.render_anchor

    def changed(env, step, *args):
        image = original(env, step, *args)
        if step == 48:
            image["current_endpoint"]["sim_state_sha256"] = "a" * 64
        return image

    case.observers.render_anchor = changed
    with pytest.raises(ValueError, match="without changing the simulation"):
        capture(case, receipt, options)
    assert "reobserve_step48" not in receipt
