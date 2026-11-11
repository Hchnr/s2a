"""Only a still-current endpoint may replace a repeated observation."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from s2a.research.wam.training.native.native_training import state_sha256


def environment():
    data = SimpleNamespace(time=6.0, qpos=np.array([1.0, 2.0]), qvel=np.array([3.0, 4.0]))
    sim = SimpleNamespace(data=data)
    sim.get_state = lambda: SimpleNamespace(flatten=lambda: np.r_[data.time, data.qpos, data.qvel])
    return SimpleNamespace(sim=sim)


def test_same_endpoint_is_reused_without_a_second_state_mutating_read():
    env, calls = environment(), []

    def endpoint(actual, step):
        calls.append(step)
        if len(calls) > 1:
            actual.sim.data.qpos[1] += 0.1
        state = actual.sim.get_state().flatten()
        return {"sim_states": state, "opening": 0.4}, {
            "endpoint_step": step,
            "sim_time": actual.sim.data.time,
            "sim_state_sha256": state_sha256(state),
        }

    cache = SameStateEndpointCache(endpoint)
    first, _receipt = cache(env, 120)
    first["sim_states"][2] = 999
    second, receipt = cache(env, 120)
    assert calls == [120] and env.sim.data.qpos[1] == 2
    assert second["sim_states"][2] == 2 and receipt["endpoint_step"] == 120
    assert [event["kind"] for event in cache.events] == ["observed", "reused"]


@pytest.mark.parametrize("change", ["position", "velocity", "time"])
def test_any_state_change_rejects_cached_endpoint_without_another_read(change):
    env, calls = environment(), []

    def endpoint(actual, step):
        calls.append(step)
        state = actual.sim.get_state().flatten()
        return {"sim_states": state}, {
            "endpoint_step": step,
            "sim_time": actual.sim.data.time,
            "sim_state_sha256": state_sha256(state),
        }

    cache = SameStateEndpointCache(endpoint)
    cache(env, 120)
    if change == "position":
        env.sim.data.qpos[0] += 1e-15
    elif change == "velocity":
        env.sim.data.qvel[0] += 1e-15
    else:
        env.sim.data.time += 1e-15
    with pytest.raises(ValueError, match="exact same current state"):
        cache(env, 120)
    assert calls == [120]


def test_new_steps_are_observed_once_and_old_steps_are_rejected():
    env, calls = environment(), []

    def endpoint(actual, step):
        calls.append(step)
        state = actual.sim.get_state().flatten()
        return {"sim_states": state}, {
            "endpoint_step": step,
            "sim_time": actual.sim.data.time,
            "sim_state_sha256": state_sha256(state),
        }

    cache = SameStateEndpointCache(endpoint)
    cache(env, 120)
    env.sim.data.time += 0.05
    cache(env, 121)
    with pytest.raises(ValueError, match="must not go backwards"):
        cache(env, 120)
    assert calls == [120, 121]


def test_endpoint_with_unbound_returned_state_is_rejected():
    env = environment()

    def bad_endpoint(actual, step):
        state = actual.sim.get_state().flatten()
        return {"sim_states": state}, {"endpoint_step": step, "sim_time": actual.sim.data.time, "sim_state_sha256": "x"}

    cache = SameStateEndpointCache(bad_endpoint)
    with pytest.raises(ValueError, match="bind the actual"):
        cache(env, 120)
    assert not cache.events
