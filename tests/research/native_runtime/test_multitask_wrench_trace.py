"""Verify reset and subsequent wrench observations retain causal endpoint identity."""

from types import SimpleNamespace

import pytest

from scripts.simulation.runtime.replay_multitask_wrench import CurrentWrenchTrace


def test_trace_includes_reset_and_rejects_changed_repeated_state():
    trace = CurrentWrenchTrace(1, reader=lambda _: {"time": 0.0, "readonly_verified": True}, contacts=lambda _: [])
    receipt = {"endpoint_step": 0, "sim_time": 0.0, "sim_state_sha256": "reset"}
    observers = SimpleNamespace(endpoint=None, current=None, render_anchor=None, render_cameras=None)
    trace.wrap(None, {"endpoint_receipt": receipt}, observers)
    assert trace.rows[0]["sim_state_sha256"] == "reset"
    with pytest.raises(ValueError, match="identity changed"):
        trace.capture(None, 0, {**receipt, "sim_state_sha256": "different"})
    with pytest.raises(ValueError, match="registered completed endpoints"):
        trace.capture(None, 2, {**receipt, "endpoint_step": 2})
