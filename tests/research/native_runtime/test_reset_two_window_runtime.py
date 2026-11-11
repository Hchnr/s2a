"""A two-window reset consumes actual first-window history and fixed budget."""

from pathlib import Path

import numpy as np
import pytest

from s2a.research.native_runtime.reset_stream_boundary import first_window_inner
from s2a.research.native_runtime.reset_two_window_runtime import ResetTwoWindowRuntime
from tests.research.native_runtime.test_reset_endpoint import sample


def test_reset_continuation_has_no_prior_command_and_sixteen_period_budget():
    expected, endpoint, chunk, controller = sample()
    runtime = ResetTwoWindowRuntime(
        expected=expected, expected_receipt=endpoint, chunk=chunk, controller=controller, tool=np.eye(4)
    )

    def callback(*_args, **_kwargs):
        return None

    def fake_run(*args, **kwargs):
        assert len(args[3]) == 0 and args[3][-1] is None
        assert kwargs["continuation_budget"] == kwargs["continuation_limit"] == 16
        assert kwargs["continuation"] is callback and kwargs["registered_seed"] == 42
        return {"behavioural_replay_verified": True}

    runtime._run = fake_run
    result = runtime.run_feedback({}, {}, "", Path("."), {}, registered_seed=42, continuation=callback)
    assert result["reset_observation_verified"] is True and result["behavioural_replay_verified"] is False
    with pytest.raises(ValueError, match="continuation"):
        runtime.run_feedback({}, {}, "", Path("."), {}, registered_seed=42, continuation=None)


def test_boundary_uses_actual_inner_samples_and_rejects_incomplete_or_changed_periods():
    limits = np.tile([-1.0, 1.0], (7, 1))
    inner = [dict(time=0.4 + 0.002 * (i + 1), q=[0.1] * 7, qvel=[0] * 7) for i in range(25)]
    periods = [dict(step=i) for i in range(8)]
    periods[-1].update(
        actual_inner=inner, endpoint_receipt={"sim_time": 0.45}, inner_below_buffer_samples=0, inner_outside_samples=0
    )
    boundary = first_window_inner(periods, limits)
    assert boundary["prefix_step"] == 7 and boundary["minimum_margin_rad"] == pytest.approx(0.9)
    inner[0]["q"][0] = 0.99
    assert boundary["samples"][0]["q"][0] == 0.1
    with pytest.raises(ValueError, match="safety counts"):
        first_window_inner(periods, limits)
    with pytest.raises(ValueError, match="Eight actual"):
        first_window_inner(periods[:-1], limits)
