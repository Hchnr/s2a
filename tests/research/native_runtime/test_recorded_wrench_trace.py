"""Check endpoint timing and repeat handling for the recorded wrench replay."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime.inspect_recorded_wrench import EndpointTrace


def test_trace_rejects_wrong_endpoint_and_records_only_completed_observations():
    data = SimpleNamespace(ctrl=np.array([0.4]), actuator_force=np.array([0.5]), qfrc_actuator=np.array([0.6]))
    env = SimpleNamespace(sim=SimpleNamespace(data=data))
    trace = EndpointTrace(
        82,
        wrench_reader=lambda _env: {"time": 1.0, "readonly_verified": True},
        contact_reader=lambda _env: [],
    )
    endpoint = {"endpoint_step": 81, "sim_time": 1.0, "sim_state_sha256": "state"}
    trace.capture(env, 81, endpoint)
    trace.capture(env, 81, endpoint)
    data.ctrl[0] = 9.0
    assert trace.rows[81]["ctrl"][0] == 0.4
    assert trace.repeated == 1
    with pytest.raises(ValueError, match="completed recorded continuation"):
        trace.capture(env, 83, endpoint)
    with pytest.raises(ValueError, match="changed state or time"):
        trace.capture(env, 81, {**endpoint, "sim_state_sha256": "other"})
    with pytest.raises(ValueError, match="timestamp-bound"):
        trace.capture(env, 82, {**endpoint, "endpoint_step": 82, "sim_time": 2.0})
