"""The oracle trial retains real denominators and causal segment boundaries."""

import numpy as np
import pytest

from s2a.research.native_runtime.correct_reference_continuation import boundary, summarize


def period(index, *, success=True):
    return dict(
        step=index,
        tcp_error=[0.003, 0.004, 0.0, 0.0, 0.0, 0.02],
        completed_feedback=dict(score=dict(opening_error=-0.1)),
        task_success=success,
        target_grasp=index % 2 == 0,
        pose_caps_satisfied=True,
        minimum_inner_margin_rad=0.2,
        inner_below_buffer_samples=0,
        inner_outside_samples=0,
    )


def test_stability_requires_all_registered_periods_and_every_final_predicate():
    rows = [period(index) for index in range(80)]
    result = summarize(rows, expected_periods=80)
    assert result["full_trajectory"] and result["stable_last_40"]
    assert result["successful_periods"] == 80 and result["grasp_periods"] == 40
    np.testing.assert_allclose(
        [result["position_rmse_m"], result["rotation_rmse_rad"], result["opening_rmse"]], [0.005, 0.02, 0.1]
    )
    assert not summarize(rows[:60], expected_periods=80)["stable_last_40"]
    rows[-2]["task_success"] = False
    assert not summarize(rows, expected_periods=80)["stable_last_40"]


def test_summary_rejects_skipped_periods_or_nonfinite_measurements():
    rows = [period(0), period(2)]
    with pytest.raises(ValueError, match="consecutive"):
        summarize(rows, expected_periods=80)
    rows = [period(0)]
    rows[0]["tcp_error"][0] = float("nan")
    with pytest.raises(ValueError, match="Finite measured TCP"):
        summarize(rows, expected_periods=80)


def test_refresh_and_final_stop_require_completed_causal_periods():
    refresh = dict(
        steps_completed=3, stop=dict(reason="requested_observation", at_step=11, next_action_dispatched=False)
    )
    assert boundary(refresh, start=8, total=19) == "quality_trigger"
    refresh["steps_completed"] = 0
    with pytest.raises(ValueError, match="completed current period"):
        boundary(refresh, start=8, total=19)
    end = dict(
        steps_completed=3, stop=dict(reason="oracle_reference_complete", at_step=19, next_action_dispatched=False)
    )
    assert boundary(end, start=16, total=19) == "reference_complete"
    end["stop"]["at_step"] = 18
    with pytest.raises(ValueError, match="registered final period"):
        boundary(end, start=16, total=19)
