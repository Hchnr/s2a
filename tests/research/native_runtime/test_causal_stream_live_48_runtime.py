"""Exercise long-stream control and checker boundaries without physics or GPU."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.simulation.runtime import run_causal_stream_live_48 as runtime


class Channel:
    def __init__(self, *, fail=False):
        self.frames = []
        self.fail = fail

    def send(self, frame):
        if self.fail:
            raise OSError("synthetic send failure")
        self.frames.append(frame)


def ledger(count, step):
    return SimpleNamespace(issued_requests=count, session="registered", current={"step": step})


def stopped(count, step):
    return {
        "complete": False,
        "requests_issued": count,
        "stop": {"reason": "quality_no_witness", "at_step": step, "next_action_dispatched": False},
    }


@pytest.mark.parametrize(("count", "step"), [(0, 80), (1, 83), (2, 96), (5, 119)])
def test_safe_stop_sends_only_session_and_actual_endpoint(count, step):
    channel, receipt = Channel(), {}
    runtime.close_model_stream(channel, ledger(count, step), stopped(count, step), receipt)
    assert channel.frames == [{"version": 1, "kind": "stop", "session": "registered", "next_step": step}]
    assert receipt["stream_stop_frame"]["sent"] is True
    assert "quality_no_witness" not in str(channel.frames)


def test_stop_after_last_response_does_not_send_to_a_closed_model():
    channel, receipt = Channel(), {}
    runtime.close_model_stream(channel, ledger(6, 125), stopped(6, 125), receipt)
    assert channel.frames == []
    assert receipt["stream_stop_frame"] == {"required": False, "reason": "model_request_budget_exhausted"}


def test_failed_stop_send_preserves_the_attempt():
    channel, receipt = Channel(fail=True), {}
    with pytest.raises(OSError, match="synthetic send failure"):
        runtime.close_model_stream(channel, ledger(1, 83), stopped(1, 83), receipt)
    assert receipt["stream_stop_frame"]["send_attempted"] is True
    assert receipt["stream_stop_frame"]["sent"] is False


def test_later_windows_do_not_access_historical_trajectories(monkeypatch):
    def forbidden(*_args):
        raise AssertionError("a future historical observation was read")

    monkeypatch.setattr(runtime.historical, "guard_period", forbidden)
    monkeypatch.setattr(runtime.historical, "gate_response", forbidden)
    assert runtime.guard_period(None, {"step": 96}) is True
    assert runtime.gate_response(None, {"request_id": 2}, {}) is True
    with pytest.raises(AssertionError, match="future historical"):
        runtime.guard_period(None, {"step": 95})
    with pytest.raises(AssertionError, match="future historical"):
        runtime.gate_response(None, {"request_id": 1}, {})


def test_checker_entry_does_not_import_the_candidate_policy():
    code = """
from scripts.simulation.runtime import run_causal_stream_live_48 as runtime
from s2a.research.native_runtime.streaming_replay import replay_streaming_continuation
runtime.prior.checker_isolation()
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr
