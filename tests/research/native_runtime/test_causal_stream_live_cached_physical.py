"""Keep the frozen long-stream workers while wrapping endpoint observations."""

import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import run_causal_stream_live_48 as frozen
from scripts.simulation.runtime import run_causal_stream_live_cached as cached


def test_only_the_guarded_prefix_call_differs_from_the_frozen_workers():
    for name in ("simulation_worker", "checker_worker"):
        expected = inspect.getsource(getattr(frozen, name)).replace("_run_prefix(", "_run_cached_prefix(")
        assert inspect.getsource(getattr(cached, name)) == expected


def test_wrapper_reuses_same_state_without_mutating_frozen_runtime(monkeypatch, tmp_path):
    data = SimpleNamespace(time=4.0, qpos=np.zeros(7), qvel=np.zeros(7))
    sim = SimpleNamespace(data=data)
    sim.get_state = lambda: SimpleNamespace(flatten=lambda: np.r_[data.time, data.qpos, data.qvel])
    env, calls = SimpleNamespace(sim=sim), []

    def endpoint(actual, step):
        calls.append(step)
        if len(calls) > 1:
            actual.sim.data.qpos[0] += 0.1
        state = actual.sim.get_state().flatten()
        return {"sim_states": state}, {
            "endpoint_step": step,
            "sim_time": actual.sim.data.time,
            "sim_state_sha256": state_sha256(state),
        }

    observers = SimpleNamespace(endpoint=endpoint, current=object(), render_anchor=object(), render_cameras=object())
    receipt = {}

    def run(_old, _directory, continuation):
        return continuation(env, {}, observers, receipt)

    def continuation(actual, _terminal, wrapped, actual_receipt, *, progress=None):
        assert wrapped.current is observers.current and wrapped.render_anchor is observers.render_anchor
        wrapped.endpoint(actual, 80)
        wrapped.endpoint(actual, 80)
        assert actual_receipt is receipt and progress is None
        return "closed"

    monkeypatch.setattr(frozen, "_run_prefix", run)
    assert cached._run_cached_prefix({}, tmp_path, continuation) == "closed"
    assert calls == [80] and env.sim.data.qpos[0] == 0
    assert [event["kind"] for event in receipt["endpoint_cache_events"]] == ["observed", "reused"]


def test_cached_checker_import_does_not_load_candidate_policies():
    code = """
from scripts.simulation.runtime import run_causal_stream_live_cached as runner
runner.prior.checker_isolation()
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr
