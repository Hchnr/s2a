"""Owned worker results report actual exit and timeout cleanup."""

import sys

from s2a.infrastructure.owned_processes import run_owned_workers


def test_workers_are_waited_and_failures_preserved(tmp_path):
    specs = [
        {"key": "ok", "argv": [sys.executable, "-c", "print('ok')"], "log": tmp_path / "ok.log"},
        {"key": "fail", "argv": [sys.executable, "-c", "raise SystemExit(3)"], "log": tmp_path / "fail.log"},
    ]
    result = run_owned_workers(specs, cwd=tmp_path, timeout_seconds=10)
    assert result["launch_error"] is None
    assert [worker["returncode"] for worker in result["workers"]] == [0, 3]
    assert all(worker["actual_wait"] and worker["reaped"] for worker in result["workers"])
    assert (tmp_path / "ok.log").read_text().strip() == "ok"


def test_timed_out_worker_is_terminated_and_reaped(tmp_path):
    spec = {"key": "slow", "argv": [sys.executable, "-c", "import time; time.sleep(30)"], "log": tmp_path / "slow.log"}
    worker = run_owned_workers([spec], cwd=tmp_path, timeout_seconds=0.1)["workers"][0]
    assert worker["signal_sent"] is True
    assert worker["actual_wait"] is worker["reaped"] is True
    assert worker["returncode"] != 0
