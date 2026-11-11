"""Exercise complete prefix recording and the pre-dispatch buffer guard."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime import development_prefix_runtime as original
from s2a.research.native_runtime.guarded_correct_prefix import (
    GuardedCorrectTask2Seed42PrefixRuntime,
    GuardedPrefixInner,
    PrefixBufferStop,
    guarded_replay,
)
from scripts.simulation.runtime.run_causal_event_feedback import check_prefix_inner


def make_env():
    model = SimpleNamespace(
        jnt_range=np.tile([-1.0, 1.0], (7, 1)),
        joint_name2id=lambda name: int(name.removeprefix("robot0_joint")) - 1,
    )
    data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7), time=0.0)
    robot = SimpleNamespace(_ref_joint_pos_indexes=np.arange(7), _ref_joint_vel_indexes=np.arange(7))
    return SimpleNamespace(robots=[robot], sim=SimpleNamespace(model=model, data=data))


def observe_period(collector, env, index, *, violate=False):
    for sample in range(25):
        env.sim.data.time += 0.002
        env.sim.data.qpos[0] = 0.98 if violate and sample == 4 else 0.1
        env.sim.data.qvel[0] = sample / 100.0
        collector.observe(env, index)


def test_every_prefix_period_is_recorded_and_independently_comparable():
    env, collector = make_env(), GuardedPrefixInner(2)
    for index in range(2):
        collector.before_dispatch(index)
        observe_period(collector, env, index)
    assert [len(period["samples"]) for period in collector.periods] == [25, 25]
    assert all(period["below_buffer_samples"] == 0 for period in collector.periods)
    assert collector.finish()["prefix_step"] == 1
    saved = deepcopy(collector.periods[0])
    check_prefix_inner(collector.periods[0], saved)
    saved["samples"][3]["qvel"][0] += 0.1
    with pytest.raises(AssertionError):
        check_prefix_inner(collector.periods[0], saved)


def test_buffer_loss_stops_before_next_environment_dispatch_and_preserves_receipt():
    env, collector = make_env(), GuardedPrefixInner(2)
    dispatched = []

    def step(command):
        dispatched.append(command)
        observe_period(collector, env, len(dispatched) - 1, violate=True)

    env.step = step
    receipt = {"control_steps_attempted": 0, "control_steps_completed": 0}

    def replay(env, commands, _saved, _receipts, _output, _observers, receipt):
        for command in commands:
            receipt["control_steps_attempted"] += 1
            env.step(command)
            receipt["control_steps_completed"] += 1

    with pytest.raises(PrefixBufferStop):
        guarded_replay(replay, collector, env, ["first", "second"], None, None, None, None, receipt)
    assert dispatched == ["first"]
    assert receipt["control_steps_completed"] == 1
    assert receipt["prefix_inner_periods"][0]["below_buffer_samples"] == 1
    assert receipt["prefix_inner_stop"] == {
        "reason": "observed_joint_buffer_loss",
        "after_step": 0,
        "at_step": 1,
        "next_action_dispatched": False,
    }
    assert env.step is step


def test_specialized_runtime_does_not_mutate_original_or_allow_unrecorded_prefix():
    run, replay, final = original.run_prefix_runtime, original.replay_saved_prefix, original.FinalPrefixInner
    runtime = GuardedCorrectTask2Seed42PrefixRuntime()
    assert runtime._run.__code__ is run.__code__
    assert (original.run_prefix_runtime, original.replay_saved_prefix, original.FinalPrefixInner) == (
        run,
        replay,
        final,
    )
    with pytest.raises(ValueError, match="full-prefix recording"):
        runtime.run(registered_seed=42, capture_final_prefix_inner=False)
