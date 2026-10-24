"""Detached CPU supervision with real child waits and immutable receipts.

This module never selects a GPU or restarts a failed command. Its public
process helpers are also used by bounded CPU tests; pipeline admission lives
in the continuation entrypoint.
"""

import contextlib
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
OBSERVED_ENV_KEYS = (*ENV_KEYS, "PYTHONPATH")
CPU_PYTHONPATH = os.pathsep.join(str(REPO / folder) for folder in ("src", "scripts"))


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "exact path/SHA reference required")
    path = Path(ref["path"])
    require(path.is_absolute() and path.is_file() and sha(path) == ref["sha256"], "artifact path/SHA changed")
    return path


def read(ref):
    path = checked(ref)
    require(path.suffix == ".json", "supervisor metadata reader refuses binary arrays")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == ref["sha256"], "JSON changed while reading")
    return json.loads(data)


def publish(path, value):
    """Atomically publish a new JSON, refusing to replace an existing name."""
    path = Path(path).resolve()
    require(path.is_relative_to(REPO / "outputs"), "receipts must stay in S2A outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link is an atomic no-clobber commit, unlike replace/rename.
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return reference(path)


def identity(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    result = dict(
        pid=pid,
        ppid=int(fields[1]),
        proc_state=fields[0],
        start_ticks=int(fields[19]),
        pgid=int(fields[2]),
        sid=int(fields[3]),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    result["command"] = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    if result["proc_state"] != "Z":
        result["cwd"] = str(Path(f"/proc/{pid}/cwd").resolve(strict=True))
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        environment = dict(part.split(b"=", 1) for part in raw if b"=" in part)
        result["environment"] = {key: environment.get(key.encode(), b"").decode() for key in OBSERVED_ENV_KEYS}
    return result


def cpu_environment():
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTHONPATH=CPU_PYTHONPATH,
        PYTHONDONTWRITEBYTECODE="1",
    )


def check_cpu_environment(environment):
    require(
        all(environment.get(key) == ("" if key in ENV_KEYS[:2] else "8") for key in ENV_KEYS),
        "supervisor and stage owners require explicit empty CUDA/LD and three thread limits8",
    )


def signal_owned(process, birth, signum):
    """Only signal this Popen child after checking its kernel birth and parent."""
    if process.poll() is not None:
        return False
    actual = identity(process.pid)
    require(
        actual["pid"] == birth["pid"] == process.pid
        and actual["start_ticks"] == birth["start_ticks"]
        and actual["boot_id"] == birth["boot_id"]
        and actual["ppid"] == birth["ppid"] == os.getpid(),
        "refusing to signal a process with a different birth or parent",
    )
    process.send_signal(signum)
    return True


def _stop_process(process, send, *, grace_s):
    deadline = time.monotonic() + grace_s
    signalled = killed = False
    if process.poll() is None:
        signalled = send(signal.SIGTERM)
        try:
            # Both waits share one deadline. Reserve a small final interval
            # for escalation/reap rather than doubling the cleanup budget.
            process.wait(timeout=max(0, deadline - time.monotonic() - min(2, grace_s / 4)))
        except subprocess.TimeoutExpired:
            killed = send(signal.SIGKILL)
            process.wait(timeout=max(0, deadline - time.monotonic()))
    else:
        process.wait()
    return dict(signal_sent=signalled, kill_sent=killed, reaped=True, returncode=process.returncode)


def stop_owned(process, birth, *, grace_s=2):
    """Bounded exceptional cleanup; normal wait is still required separately."""
    return _stop_process(process, lambda sig: signal_owned(process, birth, sig), grace_s=grace_s)


def _stop_before_birth(process, pidfd, *, grace_s):
    """A failed birth read cannot strand a still-owned newly created child.

    A retained pidfd, or the unreaped kernel child relationship proved using
    WNOWAIT, supplies authority without inventing a start-ticks observation.
    """

    def send(signum):
        if process.poll() is not None:
            return False
        if pidfd is not None:
            signal.pidfd_send_signal(pidfd, signum)
        else:
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            process.send_signal(signum)
        return True

    return _stop_process(process, send, grace_s=grace_s)


def check_sources(mapping):
    require(isinstance(mapping, dict) and bool(mapping), "supervisor source closure is empty")
    for path, digest in mapping.items():
        checked(dict(path=path, sha256=digest))


def owned_stage(
    command,
    *,
    environment,
    directory,
    timeout_s,
    source_sha256,
    check_result=None,
    cleanup_grace_s=2,
    observe_cleanup=None,
):
    """Own exactly one new command through actual wait and exceptional cleanup."""
    check_cpu_environment(environment)
    require(isinstance(command, list) and command and all(isinstance(x, str) and x for x in command), "invalid argv")
    require(isinstance(timeout_s, (int, float)) and timeout_s > 0, "positive command deadline required")
    require(isinstance(cleanup_grace_s, (int, float)) and cleanup_grace_s > 0, "positive cleanup limit required")
    check_sources(source_sha256)
    directory = Path(directory).resolve()
    require(directory.is_relative_to(REPO / "outputs"), "owned stage escaped S2A outputs")
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version="s2a.durable-stage-owner.v1",
        complete=False,
        passed=False,
        command=command,
        timeout_s=timeout_s,
        cleanup_grace_s=cleanup_grace_s,
        source_sha256=source_sha256,
        automatic_retry=False,
        timed_out=False,
        only_owned_child_signalled=True,
    )
    process = birth = pidfd = None
    started = time.monotonic()
    deadline = started + timeout_s
    failure = None
    try:
        publish(directory / "intent.json", dict(result, owner=identity(os.getpid()), observed_utc=utc()))
        with (directory / "command.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            if hasattr(os, "pidfd_open"):
                pidfd = os.pidfd_open(process.pid, 0)
            birth = identity(process.pid)
            require(birth["ppid"] == os.getpid(), "new stage is not this owner's kernel child")
            require(birth["pgid"] == birth["sid"] == process.pid, "new stage did not detach its session")
            result["launch_ref"] = publish(
                directory / "launch.json",
                dict(
                    child=birth,
                    owner=identity(os.getpid()),
                    command=command,
                    selected_environment={key: environment.get(key) for key in OBSERVED_ENV_KEYS},
                    source_sha256=source_sha256,
                    stdin="DEVNULL",
                    stdout=str(directory / "command.log"),
                    start_new_session=True,
                    automatic_retry=False,
                ),
            )
            code = process.wait(timeout=max(0, deadline - time.monotonic()))
            result["exit_ref"] = publish(
                directory / "exit.json",
                dict(returncode=code, actual_wait=True, automatic_retry=False, observed_utc=utc()),
            )
            require(code == 0, f"owned stage exited{code}; no restart")
            if check_result is not None:
                result["stage_result_ref"] = check_result()
            check_sources(source_sha256)
            result.update(complete=True, passed=True)
    except BaseException as error:
        failure = error
        result["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
        result["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        if process is not None:
            if birth is None:
                try:
                    release = _stop_before_birth(process, pidfd, grace_s=cleanup_grace_s)
                    release.update(
                        pid=process.pid,
                        start_ticks=None,
                        birth_capture_available=False,
                        ownership_basis="pidfd" if pidfd is not None else "unreaped kernel child wait relationship",
                        original_error=None if failure is None else repr(failure),
                        only_owned_child_signalled=True,
                        actual_wait=True,
                    )
                    result["release_ref"] = publish(directory / "release.json", release)
                    result["exit_ref"] = publish(
                        directory / "exit.json",
                        dict(
                            returncode=process.returncode, actual_wait=True, automatic_retry=False, observed_utc=utc()
                        ),
                    )
                except BaseException as error:
                    result["cleanup_error"] = dict(type=type(error).__name__, message=str(error))
                    result.update(complete=False, passed=False)
            else:
                try:
                    release = stop_owned(process, birth, grace_s=cleanup_grace_s)
                    release.update(
                        pid=birth["pid"],
                        start_ticks=birth["start_ticks"],
                        original_error=None if failure is None else repr(failure),
                        only_owned_child_signalled=True,
                        actual_wait=True,
                    )
                    result["release_ref"] = publish(directory / "release.json", release)
                    if "exit_ref" not in result:
                        result["exit_ref"] = publish(
                            directory / "exit.json",
                            dict(
                                returncode=process.returncode,
                                actual_wait=True,
                                automatic_retry=False,
                                observed_utc=utc(),
                            ),
                        )
                except BaseException as error:
                    result["cleanup_error"] = dict(type=type(error).__name__, message=str(error))
                    result.update(complete=False, passed=False)
        if pidfd is not None:
            os.close(pidfd)
        if observe_cleanup is not None:
            try:
                observation = observe_cleanup()
                result["descendant_cleanup_ref"] = publish(directory / "descendant_cleanup.json", observation)
                if observation.get("all_recorded_descendants_reaped") is not True:
                    result.update(complete=False, passed=False)
            except BaseException as error:
                result["descendant_cleanup_error"] = dict(type=type(error).__name__, message=str(error))
                result.update(complete=False, passed=False)
        result["seconds"] = time.monotonic() - started
        result["result_ref_path"] = str(directory / "result.json")
        publish(directory / "result.json", result)
    require(result["passed"], f"owned stage failed; see{directory / 'result.json'}")
    return reference(directory / "result.json")


def reject_duplicate(directory, *, command):
    receipt = Path(directory) / "rejected_launches" / f"{uuid.uuid4().hex}.json"
    publish(
        receipt,
        dict(
            complete=True,
            passed=False,
            reason="existing durable launch directory; automatic restart forbidden",
            command=command,
            observed_utc=utc(),
            new_processes=0,
        ),
    )
    raise FileExistsError(f"durable launch already exists; rejection:{receipt}")


def launch_detached(command, *, directory, source_sha256, environment=None, ready_timeout_s=10):
    """Launch one independent supervisor and return after its bounded readiness.

    The detached program publishes ready.json after checking launch.json, then
    keeps its own Popen children. This launcher does not own their model waits.
    """
    environment = cpu_environment() if environment is None else environment
    check_cpu_environment(environment)
    check_sources(source_sha256)
    directory = Path(directory).resolve()
    require(directory.is_relative_to(REPO / "outputs"), "durable launch escaped S2A outputs")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        reject_duplicate(directory, command=command)
    process = birth = pidfd = None
    try:
        publish(
            directory / "launch_intent.json",
            dict(command=command, launcher=identity(os.getpid()), source_sha256=source_sha256, automatic_retry=False),
        )
        with (directory / "supervisor.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        if hasattr(os, "pidfd_open"):
            pidfd = os.pidfd_open(process.pid, 0)
        birth = identity(process.pid)
        launch_ref = publish(
            directory / "launch.json",
            dict(
                child=birth,
                command=command,
                selected_environment={key: environment.get(key) for key in OBSERVED_ENV_KEYS},
                source_sha256=source_sha256,
                stdin="DEVNULL",
                stdout=str(directory / "supervisor.log"),
                start_new_session=True,
                automatic_retry=False,
                wait_responsibility="detached supervisor owns its stage children",
            ),
        )
        deadline = time.monotonic() + ready_timeout_s
        while not (directory / "ready.json").exists():
            require(process.poll() is None, "detached supervisor exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("detached supervisor readiness timed out; no automatic restart")
            time.sleep(0.01)
        ready_ref = reference(directory / "ready.json")
        ready = read(ready_ref)
        actual = ready["identity"]
        require(
            ready["launch_ref"] == launch_ref
            and actual["pid"] == birth["pid"]
            and actual["start_ticks"] == birth["start_ticks"]
            and actual["sid"] == actual["pgid"] == birth["pid"],
            "detached readiness does not bind this launched birth/session",
        )
        return publish(
            directory / "launcher_result.json",
            dict(
                complete=True,
                passed=True,
                launch_ref=launch_ref,
                ready_ref=ready_ref,
                automatic_retry=False,
                stage_completion_claim=False,
                observed_utc=utc(),
            ),
        )
    except BaseException as error:
        result = dict(complete=False, passed=False, error=dict(type=type(error).__name__, message=str(error)))
        if process is not None:
            result["owned_cleanup"] = (
                stop_owned(process, birth) if birth is not None else _stop_before_birth(process, pidfd, grace_s=2)
            )
        with contextlib.suppress(FileExistsError):
            publish(directory / "launcher_result.json", result)
        raise
    finally:
        if pidfd is not None:
            os.close(pidfd)


def accept_detached_launch(directory, *, command, source_sha256, timeout_s=10):
    """A detached run is admitted once by its exact recorded PID and birth."""
    directory = Path(directory)
    deadline = time.monotonic() + timeout_s
    while not (directory / "launch.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("detached run has no real launcher receipt")
        time.sleep(0.01)
    launch_ref = reference(directory / "launch.json")
    launch = read(launch_ref)
    actual = identity(os.getpid())
    require(
        launch["command"] == command
        and launch["child"]["pid"] == actual["pid"]
        and launch["child"]["start_ticks"] == actual["start_ticks"]
        and launch["source_sha256"] == source_sha256
        and actual["pgid"] == actual["sid"] == actual["pid"],
        "run is not the uniquely recorded detached supervisor",
    )
    check_sources(source_sha256)
    return publish(
        directory / "ready.json",
        dict(identity=actual, launch_ref=launch_ref, source_sha256=source_sha256, observed_utc=utc()),
    )
