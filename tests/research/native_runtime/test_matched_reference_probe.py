"""First-refresh admission must not expose later recorded windows."""

import pytest

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
