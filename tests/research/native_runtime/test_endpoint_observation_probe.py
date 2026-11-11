"""Endpoint instrumentation preserves and exposes the actual state change."""

from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.native_runtime.endpoint_observation_probe import EndpointObservationProbe


def environment():
    data = SimpleNamespace(time=6.0, qpos=np.array([1.0, 2.0]), qvel=np.array([3.0, 4.0]))
    model = SimpleNamespace(nq=2, nv=2, na=0)
    sim = SimpleNamespace(data=data, model=model)
    sim.get_state = lambda: SimpleNamespace(flatten=lambda: np.r_[data.time, data.qpos, data.qvel])
    return SimpleNamespace(sim=sim)


def test_probe_records_raw_before_and_after_without_restoring_state():
    env = environment()

    def endpoint(actual, step):
        actual.sim.data.qpos[1] += 0.125
        return {"step": step}, {}

    probe = EndpointObservationProbe(endpoint, step=120)
    assert probe(env, 120)[0] == {"step": 120}
    event = probe.events[0]
    np.testing.assert_array_equal(event["changed_indices"], [2])
    assert event["max_absolute_difference"] == 0.125 and not event["time_changed"]
    assert event["before"]["sha256"] != event["after"]["sha256"]
    assert env.sim.data.qpos[1] == 2.125 and event["completed"]


def test_probe_records_a_mutation_even_if_endpoint_raises():
    env = environment()

    def endpoint(actual, _step):
        actual.sim.data.qvel[0] = 7.0
        raise RuntimeError("endpoint failed")

    probe = EndpointObservationProbe(endpoint, step=120)
    with pytest.raises(RuntimeError, match="endpoint failed"):
        probe(env, 120)
    assert not probe.events[0]["completed"]
    np.testing.assert_array_equal(probe.events[0]["changed_indices"], [3])
    assert env.sim.data.qvel[0] == 7.0


def test_other_steps_are_forwarded_without_recording():
    env = environment()
    probe = EndpointObservationProbe(lambda _env, index: index, step=120)
    assert probe(env, 119) == 119 and probe.events == []
