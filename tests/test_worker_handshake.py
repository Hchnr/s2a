"""Real CPU3.11 child admission and failure cleanup; no simulator/model imports."""

import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from s2a.worker_handshake import (
    EXPECTED_ENVIRONMENT,
    birth,
    capture_cpu_identity,
    contract_identity,
    cpu_environment,
    digest,
    go_record,
    owned_child,
    stop_and_reap,
    validate_ready,
    wait_ready,
    write_once,
)

REPO = Path(__file__).resolve().parents[1]
PHYSICS_PYTHON = REPO / "outputs/libero_env311/bin/python"
CHILD = """
import json,sys
from pathlib import Path
from s2a.worker_handshake import self_ready,write_once,wait_for_go
folder=Path(sys.argv[1])
ready=self_ready(token='only-launch',config_path=folder/'config.json',snapshot_path=folder/'snapshot.json')
write_once(folder/'ready.json',ready)
wait_for_go(folder/'ready.json',folder/'go.json',timeout_s=2)
write_once(folder/'work.json',{'calls':1,'python':list(sys.version_info[:2])})
"""


@pytest.fixture
def child(tmp_path):
    assert PHYSICS_PYTHON.exists(), "the registered CPU3.11 physics interpreter is required"
    for name in ("config.json", "snapshot.json"):
        write_once(tmp_path / name, {"test_only": True})
    command = [str(PHYSICS_PYTHON), "-c", CHILD, str(tmp_path)]
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=dict(os.environ, **EXPECTED_ENVIRONMENT, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO / "src")),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    owned = birth(process.pid, allow_exited=True)
    try:
        yield process, owned, tmp_path, command
    finally:
        stop_and_reap(process, owned)


def admitted_ready(child):
    process, owned, folder, command = child
    ready = wait_ready(process, folder / "ready.json")
    observations = []
    observed = capture_cpu_identity(
        process.pid,
        expected_ticks=owned["start_ticks"],
        expected_ppid=os.getpid(),
        expected_command=command,
        expected_cwd=REPO,
        observations=observations,
    )
    parent_identity = {key: birth(os.getpid())[key] for key in ("pid", "start_ticks")}
    validate_ready(
        ready,
        observed,
        token="only-launch",
        config_path=folder / "config.json",
        snapshot_path=folder / "snapshot.json",
        parent_identity=parent_identity,
    )
    return ready, observed, parent_identity


def registry(child, observed):
    folder = child[2]
    write_once(
        folder / "process.json",
        {"identity": contract_identity(observed), "ready_sha256": digest(folder / "ready.json")},
    )


def test_cpu311_child_cannot_work_before_go_and_executes_once(child):
    _, observed, _ = admitted_ready(child)
    process, _, folder, _ = child
    time.sleep(0.05)
    assert not (folder / "work.json").exists() and process.poll() is None
    registry(child, observed)
    write_once(folder / "go.json", go_record(folder / "ready.json", folder / "process.json"))
    assert process.wait(timeout=5) == 0
    assert json.loads((folder / "work.json").read_text()) == {"calls": 1, "python": [3, 11]}


@pytest.mark.parametrize(
    "field",
    [
        "pid",
        "start_ticks",
        "ppid",
        "command",
        "environment",
        "token",
        "config_sha256",
        "snapshot_sha256",
        "parent_identity",
    ],
)
def test_ready_identity_source_and_parent_mismatch_cannot_admit(child, field):
    ready, observed, parent = admitted_ready(child)
    wrong = copy.deepcopy(ready)
    if field in ("pid", "start_ticks", "ppid"):
        wrong["identity"][field] += 1
    elif field == "command":
        wrong["identity"][field] = ["wrong"]
    elif field == "environment":
        wrong["identity"][field]["CUDA_VISIBLE_DEVICES"] = "6"
    elif field == "parent_identity":
        wrong[field]["start_ticks"] += 1
    else:
        wrong[field] = "wrong"
    with pytest.raises(ValueError):
        validate_ready(
            wrong,
            observed,
            token="only-launch",
            config_path=child[2] / "config.json",
            snapshot_path=child[2] / "snapshot.json",
            parent_identity=parent,
        )
    assert not (child[2] / "go.json").exists() and not (child[2] / "work.json").exists()


def test_wrong_go_token_never_runs_work_and_is_reaped(child):
    _, observed, _ = admitted_ready(child)
    registry(child, observed)
    folder = child[2]
    go = go_record(folder / "ready.json", folder / "process.json")
    go["token"] = "stale"
    write_once(folder / "go.json", go)
    assert child[0].wait(timeout=5) != 0
    assert not (folder / "work.json").exists()
    assert stop_and_reap(child[0], child[1])["reaped"]


def test_no_go_timeout_and_parent_failure_both_reap_child(child):
    admitted_ready(child)
    assert child[0].wait(timeout=5) != 0
    assert not (child[2] / "work.json").exists()
    assert stop_and_reap(child[0], child[1])["reaped"]


def test_go_write_failure_reaps_the_same_owned_birth(child):
    admitted_ready(child)
    write_once(child[2] / "go.json", {"bad": True})
    with pytest.raises(FileExistsError):
        write_once(child[2] / "go.json", {"replacement": True})
    result = stop_and_reap(child[0], child[1])
    assert result["reaped"] and not (child[2] / "work.json").exists()


def test_missing_actual_environment_is_not_accepted_as_empty():
    with pytest.raises(ValueError):
        cpu_environment({key: value for key, value in EXPECTED_ENVIRONMENT.items() if key != "CUDA_VISIBLE_DEVICES"})


def test_missing_proc_environment_is_bounded_unknown_and_nonempty_is_hard_failure(tmp_path):
    base = tmp_path / "123"
    base.mkdir()
    fields = ["S", "17"] + ["0"] * 17 + ["777"] + ["0"] * 5
    (base / "stat").write_text("123 (test child) " + " ".join(fields))
    (base / "cmdline").write_bytes(b"python\0cpu-only\0")
    (base / "environ").write_bytes(b"")
    (base / "cwd").symlink_to(REPO, target_is_directory=True)
    observations = []
    with pytest.raises(TimeoutError):
        capture_cpu_identity(
            123,
            expected_ticks=777,
            expected_ppid=17,
            expected_command=["python", "cpu-only"],
            expected_cwd=REPO,
            observations=observations,
            timeout_s=0.02,
            proc_root=tmp_path,
        )
    assert observations and all(o["environment"]["CUDA_VISIBLE_DEVICES"] is None for o in observations)
    assert all(o["environ_byte_count"] == 0 for o in observations)
    (base / "environ").write_bytes(
        b"CUDA_VISIBLE_DEVICES=6\0LD_LIBRARY_PATH=\0MUJOCO_GL=osmesa\0PYOPENGL_PLATFORM=osmesa\0"
    )
    observations = []
    with pytest.raises(ValueError, match="forbidden nonmissing"):
        capture_cpu_identity(
            123,
            expected_ticks=777,
            expected_ppid=17,
            expected_command=["python", "cpu-only"],
            expected_cwd=REPO,
            observations=observations,
            timeout_s=1,
            proc_root=tmp_path,
        )
    assert len(observations) == 1


def test_short_lived_cpu_child_is_reaped():
    process = subprocess.Popen(
        [str(PHYSICS_PYTHON), "-c", "raise SystemExit(7)"], env=dict(os.environ, **EXPECTED_ENVIRONMENT)
    )
    owned = birth(process.pid, allow_exited=True)
    time.sleep(0.1)
    assert stop_and_reap(process, owned)["returncode"] == 7


def test_production_registry_and_trial_builders_accept_real_command(child):
    sys.path.insert(0, str(REPO / "scripts"))
    from run_wam_temporal_continuation_matrix import matrix_trial, process_registry

    _, observed, _ = admitted_ready(child)
    folder = child[2]
    record = process_registry(observed, folder / "ready.json", folder / "snapshot.json", [observed])
    trial = matrix_trial(observed, "task2_init0", 0)
    assert record["command"] == trial["command"] == child[3]
    assert trial["returncode"] == 0 and trial["record"] == "task2_init0"


def test_production_owned_scope_reaps_after_parent_exception(tmp_path):
    process = None
    with pytest.raises(RuntimeError, match="admission failed"):
        with owned_child(
            [str(PHYSICS_PYTHON), "-c", "import time;time.sleep(10)"],
            environment=dict(os.environ, **EXPECTED_ENVIRONMENT),
            cwd=REPO,
            folder=tmp_path,
            start_new_session=True,
        ) as (process, _):
            assert os.getpgid(process.pid) == process.pid
            raise RuntimeError("admission failed")
    release = json.loads((tmp_path / "release.json").read_text())
    assert release["reaped"] and release["returncode"] == -15 and process.poll() == -15
    assert "admission failed" in release["original_error"]


def test_production_scope_reaps_child_when_parent_handles_sigterm(tmp_path):
    parent_code = """
import os,signal,sys
from pathlib import Path
from s2a.worker_handshake import owned_child
def interrupted(signum,frame):raise InterruptedError('test SIGTERM')
signal.signal(signal.SIGTERM,interrupted)
with owned_child([sys.executable,'-c','import time;time.sleep(10)'],environment=dict(os.environ),cwd=Path.cwd(),folder=Path(sys.argv[1])) as (process,owned):
 process.wait()
"""
    parent = subprocess.Popen(
        [str(PHYSICS_PYTHON), "-c", parent_code, str(tmp_path)],
        cwd=REPO,
        env=dict(os.environ, **EXPECTED_ENVIRONMENT, PYTHONPATH=str(REPO / "src")),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    owned = birth(parent.pid, allow_exited=True)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "pending.json").exists():
            assert parent.poll() is None, parent.communicate()[1].decode()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        pending = json.loads((tmp_path / "pending.json").read_text())
        parent.terminate()
        assert parent.wait(timeout=5) != 0
        release = json.loads((tmp_path / "release.json").read_text())
        assert release["pid"] == pending["pid"] and release["start_ticks"] == pending["start_ticks"]
        assert release["reaped"] and not Path(f"/proc/{pending['pid']}").exists()
    finally:
        stop_and_reap(parent, owned)


def test_birth_observation_failure_uses_kernel_pidfd_and_reaps_real_child(tmp_path, monkeypatch):
    import s2a.worker_handshake as handshake

    def broken_birth(*args, **kwargs):
        raise OSError("injected birth observation failure")

    monkeypatch.setattr(handshake, "birth", broken_birth)
    with pytest.raises(OSError, match="injected birth"):
        with handshake.owned_child(
            [str(PHYSICS_PYTHON), "-c", "import time;time.sleep(10)"],
            environment=dict(os.environ, **EXPECTED_ENVIRONMENT),
            cwd=REPO,
            folder=tmp_path,
        ):
            pytest.fail("failed birth observation must never admit a child")
    release = json.loads((tmp_path / "release.json").read_text())
    assert release["ownership_basis"] == "pidfd" and release["start_ticks"] is None
    assert release["reaped"] and not Path(f"/proc/{release['pid']}").exists()


def test_missing_proc_observation_can_recover_without_treating_unknown_as_empty(tmp_path):
    base = tmp_path / "123"
    base.mkdir()
    fields = ["S", "17"] + ["0"] * 17 + ["777"] + ["0"] * 5
    (base / "stat").write_text("123 (test child) " + " ".join(fields))
    (base / "cmdline").write_bytes(b"python\0cpu-only\0")
    (base / "environ").write_bytes(b"")
    (base / "cwd").symlink_to(REPO, target_is_directory=True)

    def publish():
        time.sleep(0.03)
        temporary = base / "complete_environment"
        temporary.write_bytes(b"CUDA_VISIBLE_DEVICES=\0LD_LIBRARY_PATH=\0MUJOCO_GL=osmesa\0PYOPENGL_PLATFORM=osmesa\0")
        temporary.replace(base / "environ")

    thread = threading.Thread(target=publish)
    thread.start()
    observations = []
    try:
        observed = capture_cpu_identity(
            123,
            expected_ticks=777,
            expected_ppid=17,
            expected_command=["python", "cpu-only"],
            expected_cwd=REPO,
            observations=observations,
            timeout_s=1,
            proc_root=tmp_path,
        )
    finally:
        thread.join()
    assert observations[0]["environment"]["CUDA_VISIBLE_DEVICES"] is None
    assert observations[-1]["environment"] == observed["environment"] == EXPECTED_ENVIRONMENT


def test_cpu311_missing_birth_is_reaped_using_unreaped_kernel_child_relationship(tmp_path):
    code = """
import os,sys
from pathlib import Path
import s2a.worker_handshake as h
def unavailable(*args,**kwargs):raise OSError('test unavailable proc birth')
h.birth=unavailable
try:
 with h.owned_child([sys.executable,'-c','import time;time.sleep(10)'],environment=dict(os.environ),cwd=Path.cwd(),folder=Path(sys.argv[1])):raise AssertionError('unreachable')
except OSError as exc:
 assert 'unavailable proc birth' in str(exc)
"""
    result = subprocess.run(
        [str(PHYSICS_PYTHON), "-c", code, str(tmp_path)],
        cwd=REPO,
        env=dict(os.environ, **EXPECTED_ENVIRONMENT, PYTHONPATH=str(REPO / "src")),
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode()
    release = json.loads((tmp_path / "release.json").read_text())
    assert release["reaped"] and release["start_ticks"] is None
    assert release["ownership_basis"] == "unreaped kernel child wait relationship"
    assert not Path(f"/proc/{release['pid']}").exists()
