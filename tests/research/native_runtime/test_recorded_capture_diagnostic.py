"""Diagnostic capture comparison and bounded recorded-prefix replay."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime import recorded_capture_diagnostic as diagnostic
from scripts.simulation.runtime import reset_live_two_window_replay as old_guard


def capture():
    binding = dict(sim_time=1.0, sim_state_sha256="state")
    return dict(
        frame_index=16,
        current_endpoint=dict(endpoint_step=16, **binding),
        render_before=dict(binding),
        render_after=dict(binding),
        future_rgb_frames=0,
        camera_keys=["agentview_image", "robot0_eye_in_hand_image"],
        canvas_shape=[960, 640, 3],
        canvas_dtype="uint8",
        canvas_rgb_sha256="pixels",
        canvas_png=dict(path="different paths are allowed", sha256="png"),
        current_source_pose=np.eye(4).tolist(),
        current_opening=0.5,
    )


def test_capture_difference_identifies_nested_field_without_relaxing_original_guard():
    expected, actual = capture(), capture()
    actual["canvas_png"]["path"] = "another path"
    actual["current_source_pose"][0][3] = 5e-11
    assert diagnostic.capture_differences(expected, actual)["passed"] is True
    old_guard.compare_capture(expected, actual)

    actual["render_before"]["sim_state_sha256"] = "other"
    actual["canvas_png"]["sha256"] = "other png"
    actual["current_opening"] += 2e-10
    result = diagnostic.capture_differences(expected, actual)
    assert result["passed"] is False
    assert result["differing_paths"] == [
        "render_before.sim_state_sha256",
        "canvas_png.sha256",
        "current_opening",
    ]
    with pytest.raises(ValueError):
        old_guard.compare_capture(expected, actual)


def segment(start):
    return dict(
        start_step=start,
        boundary="window_end",
        signals=[],
        continuation=dict(periods=[dict(step=step, source=[step]) for step in range(start, start + 8)]),
    )


def test_complete_window_selection_copies_prefix_and_never_reads_later_segment():
    first = segment(8)
    selected = diagnostic.complete_windows_before([first, None], capture_step=16)
    assert [window["start_step"] for window in selected] == [8]
    first["continuation"]["periods"][0]["source"].append(-1)
    assert selected[0]["periods"][0]["source"] == [8]
    malformed = segment(8)
    malformed["signals"] = [dict(replan=True)]
    with pytest.raises(ValueError, match="complete and consecutive"):
        diagnostic.complete_windows_before([malformed], capture_step=16)
    with pytest.raises(ValueError, match="without a gap"):
        diagnostic.complete_windows_before([], capture_step=16)


def test_checker_stops_after_capture_without_loading_a_future_reference(monkeypatch):
    events = []

    class Cache:
        def __init__(self, endpoint):
            self.endpoint = endpoint
            self.values = {}
            self.events = []

        def __call__(self, env, step):
            if step not in self.values:
                self.values[step] = self.endpoint(env, step)
            return self.values[step]

    def run_continuation(_env, _current, active_observers, receipt, *, start_step, expected_periods, **kwargs):
        events.append(("execute", start_step))
        periods = []
        receipt["continuation_partial"] = dict(periods=periods)
        for offset, expected in enumerate(expected_periods):
            selected = kwargs["choose_action"](None, None, None, None, offset)
            np.testing.assert_array_equal(selected["command"], expected["command"])
            period = dict(step=start_step + offset, command=selected["command"], actual_inner=[dict(q=[0] * 7)])
            periods.append(period)
            _sample, period["endpoint_receipt"] = active_observers.endpoint(_env, start_step + offset + 1)
            period["completed_feedback"] = kwargs["on_completed"](None, selected["command"], offset)
        return dict(periods=periods, steps_completed=len(periods), stop=None)

    monkeypatch.setattr(diagnostic, "SameStateEndpointCache", Cache)
    monkeypatch.setattr(diagnostic, "run_continuation", run_continuation)
    monkeypatch.setattr(diagnostic, "measured_snapshot", lambda *_args: "unchanged")
    poses = np.tile(np.eye(4), (40, 1, 1))
    openings = np.full(40, 0.5)
    expected = [dict(step=step, command=[0] * 7, completed_feedback=dict(score="score")) for step in range(16)]
    case = dict(chunk=dict(effective_targets_tcp=poses[:8], openings=openings[:8]), tool=np.eye(4), target_object="x")

    def load(_env, _current, _index, step):
        events.append(("load", step))
        return poses, openings

    def capture_final(_env, _current, step):
        events.append(("capture", step))
        return dict(frame_index=step)

    endpoint_reads = []

    def endpoint(_env, step):
        endpoint_reads.append(step)
        return dict(source_pose=np.eye(4)), dict(endpoint_step=step)

    result = diagnostic.replay_until_capture(
        object(),
        dict(endpoint_receipt=dict(endpoint_step=0)),
        SimpleNamespace(endpoint=endpoint, current=None, render_anchor=None, render_cameras=None),
        {},
        phase="checker",
        case=case,
        initial_periods=copy.deepcopy(expected[:8]),
        windows=[dict(start_step=8, periods=copy.deepcopy(expected[8:]))],
        capture_step=16,
        load_window=load,
        capture_final=capture_final,
        score_gripper=lambda *_args: "score",
        compare_score=lambda actual, prior: np.testing.assert_equal(actual, prior),
    )
    assert events == [("execute", 0), ("load", 8), ("execute", 8), ("capture", 16)]
    assert endpoint_reads == list(range(1, 17))
    assert result["next_action_dispatched"] is False
    assert result["capture"]["frame_index"] == 16
