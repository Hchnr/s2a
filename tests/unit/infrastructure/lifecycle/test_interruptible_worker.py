"""Exercise actual worker termination and its partial-receipt boundary."""

import os
import select
import signal
import subprocess
import sys
from pathlib import Path

from s2a.infrastructure.lifecycle.interruptible_worker import interruptible_sigterm


def test_handler_is_restored_after_normal_exit():
    previous = signal.getsignal(signal.SIGTERM)
    with interruptible_sigterm():
        assert signal.getsignal(signal.SIGTERM) is not previous
    assert signal.getsignal(signal.SIGTERM) is previous


def test_sigterm_allows_finally_to_publish_a_partial_receipt(tmp_path):
    receipt = tmp_path / "partial.txt"
    script = """\
import signal
import sys
from pathlib import Path
from s2a.infrastructure.lifecycle.interruptible_worker import WorkerTermination, interruptible_sigterm

try:
    with interruptible_sigterm():
        try:
            print('ready', flush=True)
            signal.pause()
        finally:
            Path(sys.argv[1]).write_text('partial')
except WorkerTermination:
    raise SystemExit(143)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(receipt)], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 30)
        assert ready and process.stdout.readline().strip() == "ready"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 143
        assert Path(receipt).read_text() == "partial"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdout is not None:
            process.stdout.close()


def test_sigkill_cannot_promise_a_receipt(tmp_path):
    receipt = tmp_path / "missing.txt"
    script = """\
import signal
import sys
from pathlib import Path
from s2a.infrastructure.lifecycle.interruptible_worker import interruptible_sigterm

with interruptible_sigterm():
    try:
        print('ready', flush=True)
        signal.pause()
    finally:
        Path(sys.argv[1]).write_text('partial')
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(receipt)], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 30)
        assert ready and process.stdout.readline().strip() == "ready"
        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=30) == -signal.SIGKILL
        assert not receipt.exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdout is not None:
            process.stdout.close()
