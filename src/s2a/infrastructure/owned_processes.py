"""Launch a bounded set of subprocesses and always wait for owned children."""

import signal
import subprocess
import time
import traceback
from pathlib import Path


def run_owned_workers(specs, *, cwd, timeout_seconds):
    """Run explicit argv/log pairs with one deadline and return actual wait results."""
    if not specs or timeout_seconds <= 0:
        raise ValueError("A nonempty worker group and positive timeout are required")
    paths = [Path(spec["log"]) for spec in specs]
    if len(set(paths)) != len(paths):
        raise ValueError("Each worker requires a distinct log")
    workers, launch_error, lifecycle = [], None, []

    def terminate_alive():
        for item in workers:
            if item["process"].poll() is None:
                try:
                    item["process"].terminate()
                    item["signaled"] = True
                except ProcessLookupError:
                    pass

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Owner received signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        try:
            for spec, path in zip(specs, paths, strict=True):
                path.parent.mkdir(parents=True, exist_ok=True)
                log = path.open("x")
                try:
                    proc = subprocess.Popen(spec["argv"], cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                workers.append({"spec": spec, "process": proc, "log": log})
        except BaseException:
            launch_error = traceback.format_exc()
            terminate_alive()
        deadline = time.monotonic() + timeout_seconds
        for item in workers:
            proc, sent = item["process"], item.get("signaled", False)
            try:
                code = proc.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                sent = True
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    code = proc.wait()
            except BaseException:
                launch_error = traceback.format_exc()
                terminate_alive()
                sent = item.get("signaled", False)
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    code = proc.wait()
            item["log"].close()
            lifecycle.append(
                {
                    "key": item["spec"]["key"],
                    "pid": proc.pid,
                    "returncode": code,
                    "actual_wait": True,
                    "reaped": True,
                    "signal_sent": sent,
                    "log": str(item["spec"]["log"]),
                }
            )
    finally:
        terminate_alive()
        for item in workers:
            proc = item["process"]
            if proc.returncode is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if not item["log"].closed:
                item["log"].close()
        signal.signal(signal.SIGTERM, previous_handler)
    return {"workers": lifecycle, "launch_error": launch_error}
