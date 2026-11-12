"""Preserve actual partial runtime receipts when an owned worker is stopped."""

import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from s2a.research.native_runtime.development_prefix_runtime_bounded import PrefixReplayFailure
from s2a.research.native_runtime.physical_cost_bounds import physical_cost_bounds
from scripts.simulation.runtime import managed_physical_worker as worker

LIMITS = dict(control_steps=115, native_steps=2875, models=3, contexts=3)


def test_completed_worker_publishes_the_actual_runtime(tmp_path):
    path = tmp_path / "result.json"
    runtime = {"complete": True, "passed": True, "control_steps_completed": 1}
    result = worker.run_managed_physical_worker(
        phase="producer", result_path=path, execute=lambda: dict(complete=True, passed=True, runtime=runtime)
    )
    assert result["complete"] and json.loads(path.read_text())["runtime"] == runtime


def test_failed_runtime_preserves_partial_counters(tmp_path):
    path = tmp_path / "result.json"
    partial = {"complete": False, "control_steps_attempted": 1, "control_steps_completed": 0}

    def fail():
        raise PrefixReplayFailure("synthetic runtime failure", partial)

    with pytest.raises(PrefixReplayFailure):
        worker.run_managed_physical_worker(phase="producer", result_path=path, execute=fail)
    result = json.loads(path.read_text())
    assert result["runtime"] == partial and not result["complete"] and not result["termination_requested"]


def test_callback_cannot_override_registered_lifecycle_fields(tmp_path):
    path = tmp_path / "result.json"
    payload = dict(complete=True, passed=True, phase="checker", runtime=dict(complete=True, passed=True))
    with pytest.raises(ValueError, match="completed physical runtime receipt"):
        worker.run_managed_physical_worker(phase="producer", result_path=path, execute=lambda: payload)
    result = json.loads(path.read_text())
    assert result["phase"] == "producer" and not result["complete"] and not result["automatic_retry"]


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
def test_owned_signal_preserves_partial_receipt_or_unknown_cost(tmp_path, signum):
    path = tmp_path / "result.json"
    script = """\
import signal
import sys
from s2a.infrastructure.lifecycle.interruptible_worker import WorkerTermination
from s2a.research.native_runtime.development_prefix_runtime_bounded import PrefixReplayFailure
from scripts.simulation.runtime.managed_physical_worker import run_managed_physical_worker

def execute():
    try:
        print('ready', flush=True)
        signal.pause()
    except WorkerTermination as error:
        partial = {'control_steps_attempted': 8, 'control_steps_completed': 7,
                   'counters': {'mj_step': {'attempted': 198, 'completed': 197}}}
        raise PrefixReplayFailure('interrupted runtime', partial) from error

run_managed_physical_worker(phase='producer', result_path=sys.argv[1], execute=execute)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 30)
        assert ready and process.stdout.readline().strip() == "ready"
        os.kill(process.pid, signum)
        returncode = process.wait(timeout=30)
        if signum == signal.SIGTERM:
            assert returncode != 0 and path.is_file()
            result = json.loads(path.read_text())
            assert not result["complete"] and result["termination_requested"]
            bounds = physical_cost_bounds(result["runtime"], limits=LIMITS, receipt_final=True)
            assert bounds["control_steps"]["lower"] == 7 and bounds["control_steps"]["upper"] == 8
            assert bounds["native_steps"]["lower"] == 197 and bounds["native_steps"]["upper"] == 198
        else:
            assert returncode == -signal.SIGKILL and not path.exists()
            bounds = physical_cost_bounds(None, limits=LIMITS, receipt_final=False)
            assert bounds["control_steps"]["upper"] == 115 and bounds["native_steps"]["upper"] == 2875
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdout is not None:
            process.stdout.close()


def test_existing_receipt_is_never_overwritten(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("original")
    with pytest.raises(ValueError, match="fresh local worker receipt"):
        worker.run_managed_physical_worker(phase="producer", result_path=path, execute=lambda: {})
    assert Path(path).read_text() == "original"
