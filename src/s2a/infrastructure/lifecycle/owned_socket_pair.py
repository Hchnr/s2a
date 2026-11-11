"""Supervise two owned children connected by one inherited local socket pair."""

import math
import os
import signal
import socket
import subprocess
import threading
import time

from s2a.infrastructure.repository_paths import repo_path, repository_root

_NAMES = ("simulation", "model")


class OwnedSocketPairInterrupted(KeyboardInterrupt):
    """An interrupted owner that has reaped its children and retained receipts."""

    def __init__(self, result):
        super().__init__("Owned socket-pair supervision was interrupted")
        self.result = result


def run_owned_socket_pair(specs, *, timeout_seconds):
    """Run exactly two explicitly configured children and wait for each one.

    Each spec supplies ``argv``, ``env`` overrides and an exclusive repo-relative
    ``log`` under outputs. ``S2A_CHANNEL_FD`` is set to the child's inherited
    socket. The caller owns admission, device authorization and artifact checks;
    this supervisor is not itself a model, simulator or physical-safety gate.
    """
    if (
        type(specs) is not dict
        or set(specs) != set(_NAMES)
        or type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or threading.current_thread() is not threading.main_thread()
    ):
        raise ValueError("Two workers, one positive deadline and a main-thread owner are required")
    root = repository_root()
    logs = {}
    for name in _NAMES:
        spec = specs[name]
        if (
            type(spec) is not dict
            or set(spec) != {"argv", "env", "log"}
            or type(spec["argv"]) is not list
            or not spec["argv"]
            or any(type(arg) is not str or not arg for arg in spec["argv"])
            or type(spec["env"]) is not dict
            or any(type(key) is not str or type(value) is not str for key, value in spec["env"].items())
            or "S2A_CHANNEL_FD" in spec["env"]
            or not {"CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"} <= set(spec["env"])
            or spec["env"]["LD_LIBRARY_PATH"] != ""
        ):
            raise ValueError("Explicit argv, isolated log and CUDA/LD environment are required")
        log = repo_path(spec["log"], root=root)
        if not log.is_relative_to(root / "outputs") or not log.parent.is_dir() or log.exists():
            raise ValueError("Each child requires a fresh output log under the repository")
        logs[name] = log
    if logs["simulation"] == logs["model"] or specs["simulation"]["env"]["CUDA_VISIBLE_DEVICES"] != "":
        raise ValueError("Distinct logs and an empty simulation CUDA mask are required")

    workers = []
    endpoints = []
    launch_error = None
    timed_out = False
    interruption = None

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Owner received signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    deadline = time.monotonic() + timeout_seconds
    try:
        endpoints = list(socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM))
        try:
            for index, name in enumerate(_NAMES):
                spec, endpoint = specs[name], endpoints[index]
                stream = logs[name].open("x")
                try:
                    environment = dict(os.environ, **spec["env"], S2A_CHANNEL_FD=str(endpoint.fileno()))
                    process = subprocess.Popen(
                        spec["argv"],
                        cwd=root,
                        env=environment,
                        pass_fds=(endpoint.fileno(),),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    stream.close()
                    raise
                workers.append(
                    {
                        "name": name,
                        "process": process,
                        "stream": stream,
                        "actual_wait": False,
                        "signal_sent": False,
                    }
                )
                endpoint.close()
        except Exception as error:
            launch_error = f"{type(error).__name__}: {error}"
        finally:
            for endpoint in endpoints:
                endpoint.close()

        failed = launch_error is not None
        while not failed:
            for item in workers:
                process = item["process"]
                if not item["actual_wait"] and process.poll() is not None:
                    process.wait()
                    item["actual_wait"] = True
                    if process.returncode != 0:
                        failed = True
            if failed or all(item["actual_wait"] for item in workers):
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt as error:
        interruption = error
    finally:
        for endpoint in endpoints:
            endpoint.close()
        for item in workers:
            process = item["process"]
            if process.poll() is None:
                try:
                    process.terminate()
                    item["signal_sent"] = True
                except ProcessLookupError:
                    pass
        for item in workers:
            process = item["process"]
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                item["signal_sent"] = True
                process.wait()
            item["actual_wait"] = True
            item["stream"].close()
        signal.signal(signal.SIGTERM, previous_handler)
    records = [
        {
            "name": item["name"],
            "pid": item["process"].pid,
            "returncode": item["process"].returncode,
            "actual_wait": item["actual_wait"],
            "reaped": item["actual_wait"] and item["process"].returncode is not None,
            "signal_sent": item["signal_sent"],
            "log": str(logs[item["name"]].relative_to(root)),
        }
        for item in workers
    ]
    result = {
        "launch_error": launch_error,
        "timed_out": timed_out,
        "interrupted": interruption is not None,
        "workers": records,
        "passed": (
            launch_error is None
            and not timed_out
            and interruption is None
            and len(records) == 2
            and all(item["reaped"] and not item["signal_sent"] and item["returncode"] == 0 for item in records)
        ),
    }
    if interruption is not None:
        raise OwnedSocketPairInterrupted(result) from interruption
    return result
