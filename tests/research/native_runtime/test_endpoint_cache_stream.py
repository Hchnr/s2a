"""Exercise endpoint reuse through the real streaming producer and replay."""

import pytest

from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from tests.research.native_runtime.test_streaming_continuation import execute, fake_case
from tests.research.native_runtime.test_streaming_replay import cases, replay


def duplicate_read_mutates(case):
    observed, endpoint = set(), case.observers.endpoint

    def mutating(actual_env, step):
        if step in observed:
            actual_env.sim.data.qpos[0] += 0.1
        observed.add(step)
        return endpoint(actual_env, step)

    return mutating


def test_repeated_endpoint_read_is_the_only_trigger_in_the_original_runner(tmp_path):
    case = fake_case(tmp_path)
    case.observers.endpoint = duplicate_read_mutates(case)
    with pytest.raises(ValueError, match="Endpoint observation changed"):
        execute(case)
    assert case.receipt["control_steps_completed"] == 88


def test_cached_stream_keeps_exact_state_across_three_windows(tmp_path):
    case = fake_case(tmp_path)
    cache = SameStateEndpointCache(duplicate_read_mutates(case))
    case.observers.endpoint = cache
    result = execute(case)
    assert result["complete"] and result["feedback_periods_committed"] == 24
    assert case.sim.inner == 600 and case.sim.data.qpos[0] == 0
    assert sum(event["kind"] == "observed" for event in cache.events) == 24
    assert [event["step"] for event in cache.events if event["kind"] == "reused"] == [88, 96, 104]


def test_cached_independent_replay_uses_only_completed_actions(tmp_path):
    producer, checker, stream = cases(tmp_path)
    cache = SameStateEndpointCache(duplicate_read_mutates(checker))
    checker.observers.endpoint = cache
    result = replay(checker, stream)
    assert result["complete"] and result["control_steps_completed"] == 24
    assert producer.sim.inner == checker.sim.inner == 600
    assert checker.receipt["newly_selected_commands"] == 0
    assert [event["step"] for event in cache.events if event["kind"] == "reused"] == [88, 96, 104]
