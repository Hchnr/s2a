"""A longer fixed-reference run needs an explicit horizon and remains causal."""

import numpy as np
import pytest

from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.development_prefix_runtime import run_prefix_runtime
from tests.research.native_runtime.test_development_continuation import fake_runtime


def execute(*, limit=16, targets=16, expected=None, stop_after=None):
    env, observers, state_reader, task_observer, terminal = fake_runtime()
    events = []
    receipt = {"control_steps_attempted": 0, "control_steps_completed": 0, "newly_selected_commands": 0}

    def action(_q, _limits, _jacobian, _previous, offset):
        events.append(("select", offset, env.sim.inner))
        if offset == stop_after:
            return {"stop": {"reason": "quality_no_witness"}}
        return {"command": np.zeros(7), "newly_selected": True}

    def completed(_env, _command, offset):
        events.append(("completed", offset, env.sim.inner))
        return {"observed_inner_steps": env.sim.inner}

    result = run_continuation(
        env,
        terminal,
        observers,
        receipt,
        start_step=1,
        previous_command=np.zeros(7),
        targets=np.tile(np.eye(4), (targets, 1, 1)),
        tool=np.eye(4),
        target_object="target",
        choose_action=action,
        stop_on_outside=True,
        expected_periods=expected,
        state_reader=state_reader,
        task_observer=task_observer,
        on_completed=completed,
        continuation_limit=limit,
    )
    return result, receipt, events


def test_sixteen_periods_are_explicit_and_observed_before_the_next_decision():
    result, receipt, events = execute()
    assert result["steps_completed"] == receipt["control_steps_completed"] == 16
    assert events == [entry for i in range(16) for entry in (("select", i, 25 * i), ("completed", i, 25 * (i + 1)))]


def test_sixteen_periods_replay_and_detect_tampered_late_inner_sample():
    producer, _, _ = execute()
    replay, _, _ = execute(expected=producer["periods"])
    assert replay["steps_completed"] == 16
    producer["periods"][12]["actual_inner"][3]["q"][0] = 0.1
    with pytest.raises(AssertionError):
        execute(expected=producer["periods"])


def test_late_no_witness_stops_before_dispatch():
    result, receipt, events = execute(stop_after=10)
    assert result["steps_completed"] == receipt["control_steps_completed"] == 10
    assert result["stop"]["at_step"] == 11
    assert result["stop"]["next_action_dispatched"] is False
    assert events[-1] == ("select", 10, 250)


def test_default_eight_step_gate_and_reference_length_remain_strict():
    with pytest.raises(ValueError, match="already-generated targets"):
        execute(limit=8, targets=16)
    with pytest.raises(ValueError, match="already-generated targets"):
        execute(limit=16, targets=8)
    with pytest.raises(ValueError, match="eight- or sixteen-step"):
        execute(limit=9, targets=9)
    with pytest.raises(ValueError, match="bounded continuation budget"):
        run_prefix_runtime(
            None,
            None,
            None,
            np.zeros((1, 7)),
            None,
            None,
            None,
            None,
            None,
            continuation=lambda: None,
            continuation_budget=16,
        )
