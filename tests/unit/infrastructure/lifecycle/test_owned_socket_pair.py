"""The owner must observe and reap both actual socket-connected children."""

import hashlib
import json
import sys

import pytest

from s2a.infrastructure.lifecycle.owned_socket_pair import run_owned_socket_pair
from s2a.infrastructure.repository_paths import repository_root


def specs(tmp_path, simulation, model):
    base = tmp_path.relative_to(repository_root())
    environment = {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    return {
        "simulation": {"argv": simulation, "env": dict(environment), "log": str(base / "simulation.log")},
        "model": {"argv": model, "env": dict(environment), "log": str(base / "model.log")},
    }


def test_two_children_complete_two_causally_sequenced_mock_windows(tmp_path):
    data = b"synthetic current frame"
    (tmp_path / "frame.png").write_bytes(data)

    def sha(value):
        return hashlib.sha256(value).hexdigest()

    config = {
        "session": "two-child-cpu-mock",
        "initial": {"step": 80, "sim_time_s": 4.0, "state_sha256": sha(b"step80")},
        "image_root": str(tmp_path.relative_to(repository_root())),
        "task": "move the object",
        "seed": 42,
        "model_sha256": sha(b"synthetic model"),
        "max_requests": 2,
    }
    image = {"path": "frame.png", "sha256": sha(data)}
    common = """
import hashlib, json, os, socket, sys
from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission, CausalStreamLedger
config = json.loads(sys.argv[1])
channel = FramedJSONChannel(socket.socket(fileno=int(os.environ['S2A_CHANNEL_FD'])),
                            max_bytes=2*1024*1024, timeout_seconds=10)
"""
    simulation = (
        common
        + """
ledger = CausalStreamLedger(**config)
image = json.loads(sys.argv[2])
for window in range(2):
    current = ledger.current
    channel.send(ledger.issue(before_render=current, after_render=current, image=image))
    ledger.accept(channel.receive(), after_wait=ledger.current)
    for _ in range(8):
        before = ledger.current
        step = before['step'] + 1
        after = dict(step=step, sim_time_s=before['sim_time_s']+0.05,
                     state_sha256=hashlib.sha256(f'step{step}'.encode()).hexdigest())
        ledger.complete_period(before=before, after=after, inner_samples=25)
assert ledger.phase == 'done' and ledger.completed_periods == 16
channel.close()
"""
    )
    model = (
        common
        + """
admission = CausalModelAdmission(**config)
for window in range(2):
    request = channel.receive()
    inputs = admission.admit(request)
    assert inputs == {'image_bytes': b'synthetic current frame', 'task': 'move the object'}
    identity = {key: request[key] for key in ('version','session','request_id','step','sim_time_s','state_sha256','seed','model_sha256')}
    response = dict(identity, image_sha256=request['image']['sha256'], denoise_calls=35,
                    action_dim_mask=[True]*10+[False]*118, action=[[0.0]*128 for _ in range(40)],
                    normalized_action=[[0.0]*128 for _ in range(40)])
    admission.complete(request_id=window, denoise_calls=35)
    channel.send(response)
assert admission.phase == 'done'
channel.close()
"""
    )
    workers = specs(
        tmp_path,
        [sys.executable, "-c", simulation, json.dumps(config), json.dumps(image)],
        [sys.executable, "-c", model, json.dumps(config)],
    )
    result = run_owned_socket_pair(workers, timeout_seconds=15)
    assert result["passed"] and result["launch_error"] is None and not result["timed_out"]
    assert [worker["returncode"] for worker in result["workers"]] == [0, 0]
    assert all(worker["actual_wait"] and worker["reaped"] and not worker["signal_sent"] for worker in result["workers"])


def test_failed_child_stops_and_reaps_the_other_child(tmp_path):
    workers = specs(
        tmp_path,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        [sys.executable, "-c", "raise SystemExit(3)"],
    )
    result = run_owned_socket_pair(workers, timeout_seconds=10)
    assert not result["passed"] and not result["timed_out"]
    assert result["workers"][1]["returncode"] == 3
    assert result["workers"][0]["signal_sent"]
    assert all(worker["actual_wait"] and worker["reaped"] for worker in result["workers"])


def test_partial_launch_failure_stops_and_reaps_the_launched_child(tmp_path):
    workers = specs(
        tmp_path,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        [str(tmp_path / "missing-executable")],
    )
    result = run_owned_socket_pair(workers, timeout_seconds=10)
    assert not result["passed"] and result["launch_error"] is not None
    assert len(result["workers"]) == 1
    assert result["workers"][0]["actual_wait"] and result["workers"][0]["reaped"]


def test_deadline_stops_and_reaps_both_children(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    result = run_owned_socket_pair(specs(tmp_path, command, command), timeout_seconds=0.2)
    assert result["timed_out"] and not result["passed"]
    assert len(result["workers"]) == 2
    assert all(worker["signal_sent"] and worker["actual_wait"] and worker["reaped"] for worker in result["workers"])


def test_simulation_cannot_inherit_a_gpu_mask(tmp_path):
    command = [sys.executable, "-c", "pass"]
    workers = specs(tmp_path, command, command)
    workers["simulation"]["env"]["CUDA_VISIBLE_DEVICES"] = "0"
    with pytest.raises(ValueError):
        run_owned_socket_pair(workers, timeout_seconds=5)
