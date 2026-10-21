"""Identity-bound CPU worker admission, before any experiment work is allowed.

Only selected nonsecret environment values are recorded. Missing /proc values
are unknown observations, never evidence that a child inherited nonempty CUDA.
"""

import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path

ENVIRONMENT = ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "MUJOCO_GL", "PYOPENGL_PLATFORM")
EXPECTED_ENVIRONMENT = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")


def write_once(path, value):
    """Publish one complete JSON record atomically, refusing an existing target."""
    path = Path(path)
    temporary = path.with_name(path.name + f".pending.{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_environment(environment):
    """Check actual Python values; absence is rejected as well as nonempty CUDA."""
    expected = EXPECTED_ENVIRONMENT
    if any(environment.get(key) != value for key, value in expected.items()):
        raise ValueError("worker requires explicitly empty CUDA/LD and OSMesa")
    return expected


def birth(pid, *, proc_root=Path("/proc"), allow_exited=False):
    values = (Path(proc_root) / str(pid) / "stat").read_text().rpartition(")")[2].split()
    if values[0] in ("Z", "X") and not allow_exited:
        raise ProcessLookupError("owned worker exited")
    return dict(pid=pid, start_ticks=int(values[19]), ppid=int(values[1]), proc_state=values[0])


def identity(pid, *, proc_root=Path("/proc")):
    root = Path(proc_root) / str(pid)
    before = birth(pid, proc_root=proc_root)
    raw = (root / "environ").read_bytes()
    environment = dict(item.split("=", 1) for item in raw.decode().split("\0") if "=" in item)
    command = (root / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    cwd = str((root / "cwd").resolve(strict=True))
    after = birth(pid, proc_root=proc_root)
    if (before["start_ticks"], before["ppid"]) != (after["start_ticks"], after["ppid"]):
        raise ProcessLookupError("worker identity changed during observation")
    return dict(
        **after,
        command=command,
        cwd=cwd,
        environment={key: environment.get(key) for key in ENVIRONMENT},
        environ_byte_count=len(raw),
    )


def capture_cpu_identity(
    pid,
    *,
    expected_ticks,
    expected_ppid,
    expected_command,
    expected_cwd,
    observations,
    timeout_s=5.0,
    proc_root=Path("/proc"),
):
    """Retry missing observations only, while every available identity must match."""
    deadline = time.monotonic() + timeout_s
    while True:
        item = identity(pid, proc_root=proc_root)
        observations.append(item)
        if (
            item["start_ticks"] != expected_ticks
            or item["ppid"] != expected_ppid
            or item["command"] != expected_command
            or item["cwd"] != str(Path(expected_cwd).resolve())
        ):
            raise ValueError("worker birth/parent/command/cwd differs from its owned launch")
        environment = item["environment"]
        for key, value in EXPECTED_ENVIRONMENT.items():
            if environment[key] is not None and environment[key] != value:
                raise ValueError(f"worker actual {key} has a forbidden nonmissing value")
        if all(environment[key] is not None for key in ENVIRONMENT):
            cpu_environment(environment)
            return item
        if time.monotonic() >= deadline:
            raise TimeoutError("worker /proc environment remained unknown; no admission granted")
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def contract_identity(value):
    """Compare stable identity fields; transient process state is not an identity."""
    return {key: value[key] for key in ("pid", "start_ticks", "ppid", "command", "cwd", "environment")}


def self_ready(*, token, config_path, snapshot_path):
    cpu_environment(os.environ)
    item = identity(os.getpid())
    observations = []
    item = capture_cpu_identity(
        os.getpid(),
        expected_ticks=item["start_ticks"],
        expected_ppid=os.getppid(),
        expected_command=item["command"],
        expected_cwd=Path.cwd(),
        observations=observations,
    )
    return dict(
        schema_version="s2a.cpu-worker-ready.v1",
        token=token,
        identity=contract_identity(item),
        parent_identity={key: birth(os.getppid())[key] for key in ("pid", "start_ticks")},
        config_path=str(Path(config_path).resolve()),
        config_sha256=digest(config_path),
        snapshot_path=str(Path(snapshot_path).resolve()),
        snapshot_sha256=digest(snapshot_path),
        experiment_work_started=False,
        self_observations=observations,
    )


def validate_ready(ready, observed, *, token, config_path, snapshot_path, parent_identity):
    observations = ready.get("self_observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("self observations missing")
    for item in observations:
        if any(item[key] != observed[key] for key in ("pid", "start_ticks", "ppid", "command", "cwd")):
            raise ValueError("self observation changed identity")
        if any(item["environment"].get(key) not in (None, value) for key, value in EXPECTED_ENVIRONMENT.items()):
            raise ValueError("self observation recorded a forbidden environment")
    if contract_identity(observations[-1]) != contract_identity(observed):
        raise ValueError("last self observation must be complete")
    if ready != dict(
        schema_version="s2a.cpu-worker-ready.v1",
        token=token,
        identity=contract_identity(observed),
        parent_identity=parent_identity,
        config_path=str(Path(config_path).resolve()),
        config_sha256=digest(config_path),
        snapshot_path=str(Path(snapshot_path).resolve()),
        snapshot_sha256=digest(snapshot_path),
        experiment_work_started=False,
        self_observations=observations,
    ):
        raise ValueError("child-ready identity or immutable source binding differs")


def go_record(ready_path, registry_path):
    ready = json.loads(Path(ready_path).read_text())
    registry = json.loads(Path(registry_path).read_text())
    if registry["ready_sha256"] != digest(ready_path) or registry["identity"] != ready["identity"]:
        raise ValueError("process registration does not bind this ready identity")
    return dict(
        schema_version="s2a.cpu-worker-go.v1",
        token=ready["token"],
        identity=ready["identity"],
        parent_identity=ready["parent_identity"],
        ready_sha256=digest(ready_path),
        registry_path=str(Path(registry_path).resolve()),
        registry_sha256=digest(registry_path),
        config_sha256=ready["config_sha256"],
        snapshot_sha256=ready["snapshot_sha256"],
    )


def wait_for_go(ready_path, go_path, *, timeout_s=15.0):
    ready_path, go_path = Path(ready_path), Path(go_path)
    ready = json.loads(ready_path.read_text())
    deadline = time.monotonic() + timeout_s
    while not go_path.exists():
        if os.getppid() != ready["identity"]["ppid"]:
            raise ProcessLookupError("worker parent exited before admission")
        parent = birth(os.getppid())
        if {key: parent[key] for key in ("pid", "start_ticks")} != ready["parent_identity"]:
            raise ProcessLookupError("worker parent birth changed before admission")
        if time.monotonic() >= deadline:
            raise TimeoutError("parent did not admit this worker")
        time.sleep(0.01)
    go = json.loads(go_path.read_text())
    parent = birth(os.getppid())
    if {key: parent[key] for key in ("pid", "start_ticks")} != ready["parent_identity"]:
        raise ProcessLookupError("worker parent birth changed before go validation")
    registry_path = Path(go["registry_path"])
    if go != go_record(ready_path, registry_path):
        raise ValueError("go does not match ready/registration/source identities")
    if contract_identity(identity(os.getpid())) != ready["identity"]:
        raise ValueError("worker identity changed after readiness")
    if (
        digest(ready["config_path"]) != ready["config_sha256"]
        or digest(ready["snapshot_path"]) != ready["snapshot_sha256"]
    ):
        raise ValueError("worker source binding changed while awaiting admission")
    return go


def wait_ready(process, ready_path, *, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not Path(ready_path).exists():
        if process.poll() is not None:
            raise ChildProcessError(f"worker exited before readiness: {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError("worker readiness timed out")
        time.sleep(0.01)
    return json.loads(Path(ready_path).read_text())


def stop_and_reap(process, expected_birth, *, grace_s=5.0):
    """Only a live Popen child with the captured birth may receive a signal."""
    result = dict(pid=process.pid, start_ticks=expected_birth["start_ticks"], signal_sent=False)
    if process.poll() is None:
        try:
            current = birth(process.pid)
        except (ProcessLookupError, FileNotFoundError):
            current = None
        if current is not None:
            if current["start_ticks"] != expected_birth["start_ticks"] or current["ppid"] != os.getpid():
                raise ValueError("refusing to signal a child whose ownership changed")
            process.send_signal(signal.SIGTERM)
            result["signal_sent"] = True
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired as exc:
            try:
                current = birth(process.pid)
            except (ProcessLookupError, FileNotFoundError):
                current = None
            if current is not None and (
                current["start_ticks"] != expected_birth["start_ticks"] or current["ppid"] != os.getpid()
            ):
                raise ValueError("refusing to kill a child whose ownership changed") from exc
            if current is not None:
                process.kill()
            process.wait(timeout=grace_s)
            result["kill_sent"] = current is not None
    else:
        process.wait()
    result["returncode"] = process.returncode
    result["reaped"] = process.returncode is not None
    return result


@contextmanager
def owned_child(command, *, environment, cwd, folder, stream=None):
    """Shared real admission scope: failures after Popen cannot bypass reaping."""
    process = owned = None
    pidfd = None
    exception = None
    folder = Path(folder)
    try:
        process = subprocess.Popen(command, env=environment, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        if hasattr(os, "pidfd_open"):
            pidfd = os.pidfd_open(process.pid, 0)
        owned = birth(process.pid, allow_exited=True)
        write_once(folder / "pending.json", dict(**owned, command=command, automatic_retry=False))
        yield process, owned
    except BaseException as exc:
        exception = repr(exc)
        raise
    finally:
        if process is not None:
            try:
                if owned is None:
                    signalled = process.poll() is None
                    if signalled:
                        if pidfd is None:
                            # A child cannot have its PID reused while this
                            # parent retains its unreaped wait relationship.
                            # WNOWAIT proves that relationship without reaping.
                            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                            with suppress(ProcessLookupError):
                                process.terminate()
                        else:
                            with suppress(ProcessLookupError):
                                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if pidfd is None:
                            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                            with suppress(ProcessLookupError):
                                process.kill()
                        else:
                            with suppress(ProcessLookupError):
                                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                        process.wait(timeout=5)
                    cleanup = dict(
                        pid=process.pid,
                        start_ticks=None,
                        returncode=process.returncode,
                        reaped=True,
                        signal_sent=signalled,
                        birth_capture_available=False,
                        ownership_basis="pidfd" if pidfd is not None else "unreaped kernel child wait relationship",
                    )
                else:
                    cleanup = stop_and_reap(process, owned)
                write_once(
                    folder / "release.json", dict(**cleanup, original_error=exception, only_owned_child_signalled=True)
                )
            finally:
                if pidfd is not None:
                    os.close(pidfd)
