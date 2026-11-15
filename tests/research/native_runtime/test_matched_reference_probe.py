"""First-refresh admission must not expose later recorded windows."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime import matched_reference_probe as runtime
from s2a.research.native_runtime.matched_reference_probe import first_refresh_prefix


def window(start, count, *, boundary, trigger=None):
    return dict(
        start_step=start,
        continuation=dict(periods=[dict(step=step, recorded=[step]) for step in range(start, start + count)]),
        signals=[] if trigger is None else [trigger],
        boundary=boundary,
    )


def test_only_first_completed_reference_prefix_is_retained_and_copied():
    signal = dict(replan=True, current_step=18, reason="quality_margin_consumed")
    source = [
        window(8, 8, boundary="window_end"),
        window(16, 2, boundary="quality_trigger", trigger=signal),
        window(18, 8, boundary="window_end"),
    ]
    result = first_refresh_prefix(source, expected_step=18, expected_reason="quality_margin_consumed")
    assert [row["start_step"] for row in result["windows"]] == [8, 16]
    assert [row["step"] for row in result["windows"][-1]["periods"]] == [16, 17]
    source[0]["continuation"]["periods"][0]["recorded"].append("changed")
    signal["reason"] = "changed"
    assert result["windows"][0]["periods"][0]["recorded"] == [8]
    assert result["trigger"]["reason"] == "quality_margin_consumed"


def test_missing_second_point_remains_an_explicit_first_trigger():
    signal = dict(replan=True, current_step=17, reason="no_audited_second_pose")
    rows = [window(8, 8, boundary="window_end"), window(16, 1, boundary="quality_trigger", trigger=signal)]
    result = first_refresh_prefix(rows, expected_step=17, expected_reason="no_audited_second_pose")
    assert result["trigger"]["reason"] == "no_audited_second_pose"
    with pytest.raises(ValueError, match="first refresh"):
        first_refresh_prefix(rows, expected_step=17, expected_reason="quality_margin_consumed")


@pytest.mark.parametrize("change", ["gap", "early_partial", "wrong_signal_step", "wrong_boundary"])
def test_prefix_rejects_mismatched_time_or_unreported_boundary(change):
    signal = dict(replan=True, current_step=17, reason="quality_margin_consumed")
    rows = [window(8, 8, boundary="window_end"), window(16, 1, boundary="quality_trigger", trigger=signal)]
    if change == "gap":
        rows[1]["start_step"] = 17
    elif change == "early_partial":
        rows[0]["continuation"]["periods"].pop()
    elif change == "wrong_signal_step":
        signal["current_step"] = 18
    else:
        rows[1]["boundary"] = "window_end"
    with pytest.raises(ValueError):
        first_refresh_prefix(rows, expected_step=17, expected_reason="quality_margin_consumed")


def test_checker_reuses_completed_endpoints_at_reference_and_probe_boundaries(monkeypatch):
    reads = []

    class Cache:
        def __init__(self, endpoint):
            self.endpoint = endpoint
            self.values = {}
            self.events = []

        def __call__(self, env, step):
            if step not in self.values:
                self.values[step] = self.endpoint(env, step)
            return self.values[step]

    def endpoint(_env, step):
        reads.append(step)
        return dict(source_pose=np.eye(4)), dict(endpoint_step=step)

    def continuation(env, _current, observers, receipt, *, start_step, expected_periods, **kwargs):
        rows = []
        receipt["continuation_partial"] = dict(periods=rows)
        for offset, expected in enumerate(expected_periods):
            selected = kwargs["choose_action"](None, None, None, None, offset)
            _sample, endpoint_receipt = observers.endpoint(env, start_step + offset + 1)
            row = dict(
                step=start_step + offset,
                command=selected["command"],
                endpoint_receipt=endpoint_receipt,
                actual_inner=[dict(q=[0] * 7)],
            )
            rows.append(row)
            row["completed_feedback"] = kwargs["on_completed"](env, selected["command"], offset)
            np.testing.assert_array_equal(selected["command"], expected["command"])
        return dict(periods=rows, steps_completed=len(rows), stop=None)

    monkeypatch.setattr(runtime, "SameStateEndpointCache", Cache)
    monkeypatch.setattr(runtime, "run_continuation", continuation)
    monkeypatch.setattr(runtime, "measured_snapshot", lambda *_args: "unchanged")
    poses, openings = np.tile(np.eye(4), (40, 1, 1)), np.full(40, 0.5)
    expected = [dict(step=step, command=[0] * 7, completed_feedback=dict(score="score")) for step in range(9)]
    case = dict(chunk=dict(effective_targets_tcp=poses[:8], openings=openings[:8]), tool=np.eye(4), target_object="x")
    result = runtime.run_matched_reference_probe(
        object(),
        dict(endpoint_receipt=dict(endpoint_step=0)),
        SimpleNamespace(endpoint=endpoint, current=None, render_anchor=None, render_cameras=None),
        {},
        phase="checker",
        case=case,
        initial_periods=expected[:8],
        windows=[dict(start_step=8, periods=expected[8:])],
        expected_trigger=dict(current_step=9, reason="no_audited_second_pose"),
        load_window=lambda *_args: (poses, openings),
        score_gripper=lambda *_args: "score",
        compare_score=lambda actual, prior: np.testing.assert_equal(actual, prior),
        recorded_probe=dict(periods=[]),
    )
    assert reads == list(range(1, 10))
    assert result["probe"]["steps_completed"] == 0
