"""Exercise causal admission with synthetic observations, never physics or a model."""

import copy
import hashlib
import json
import os
import socket
import subprocess
import sys

import pytest

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.repository_paths import repository_root
from s2a.research.wam.tracking.runtime.causal_stream_protocol import (
    CausalModelAdmission,
    CausalStreamLedger,
    model_inputs,
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def registration(tmp_path):
    image = b"current synthetic camera frame"
    (tmp_path / "frame.png").write_bytes(image)
    return {
        "session": "cpu-mock-only",
        "initial": {"step": 80, "sim_time_s": 4.0, "state_sha256": digest(b"step80")},
        "image_root": str(tmp_path.relative_to(repository_root())),
        "task": "place the object",
        "seed": 42,
        "model_sha256": digest(b"synthetic model"),
        "max_requests": 2,
    }, {"path": "frame.png", "sha256": digest(image)}


def issue(ledger, image):
    current = ledger.current
    return ledger.issue(before_render=current, after_render=current, image=image)


def response_for(request):
    identity = {
        key: request[key]
        for key in ("version", "session", "request_id", "step", "sim_time_s", "state_sha256", "seed", "model_sha256")
    }
    return {
        **identity,
        "image_sha256": request["image"]["sha256"],
        "denoise_calls": 35,
        "action_dim_mask": [True] * 10 + [False] * 118,
        "action": [[0.0] * 128 for _ in range(40)],
        "normalized_action": [[0.0] * 128 for _ in range(40)],
    }


def complete_eight(ledger):
    for _ in range(8):
        before = ledger.current
        after = {
            "step": before["step"] + 1,
            "sim_time_s": before["sim_time_s"] + 0.05,
            "state_sha256": digest(f"step{before['step'] + 1}".encode()),
        }
        ledger.complete_period(before=before, after=after, inner_samples=25)


def test_two_sequential_windows_only_expose_current_image_and_public_task(registration):
    config, image = registration
    ledger, service = CausalStreamLedger(**config), CausalModelAdmission(**config)
    for index in range(2):
        request = issue(ledger, image)
        inputs = service.admit(request)
        assert inputs == {"image_bytes": b"current synthetic camera frame", "task": config["task"]}
        assert request["request_id"] == index
        assert request["step"] == 80 + 8 * index
        service.complete(request_id=index, denoise_calls=35)
        assert len(ledger.accept(response_for(request), after_wait=ledger.current)) == 40
        complete_eight(ledger)
    assert ledger.phase == service.phase == "done"
    assert ledger.issued_requests == ledger.accepted_requests == service.admitted_requests == 2
    assert ledger.completed_periods == 16
    assert ledger.current["step"] == 96


@pytest.mark.parametrize(
    ("key", "wrong"),
    [
        ("request_id", 1),
        ("step", 88),
        ("sim_time_s", 4.05),
        ("state_sha256", "1" * 64),
        ("image_sha256", "2" * 64),
        ("model_sha256", "3" * 64),
        ("seed", 43),
        ("denoise_calls", 34),
    ],
)
def test_wrong_response_identity_or_denoise_count_permanently_fails(registration, key, wrong):
    config, image = registration
    ledger = CausalStreamLedger(**config)
    response = response_for(issue(ledger, image))
    response[key] = wrong
    with pytest.raises(ValueError):
        ledger.accept(response, after_wait=ledger.current)
    assert ledger.phase == "failed"
    with pytest.raises(ValueError):
        issue(ledger, image)


@pytest.mark.parametrize("corruption", ["short", "nan", "inactive", "mask", "extra"])
def test_malformed_action_or_unregistered_response_field_fails(registration, corruption):
    config, image = registration
    ledger = CausalStreamLedger(**config)
    response = response_for(issue(ledger, image))
    if corruption == "short":
        response["action"].pop()
    elif corruption == "nan":
        response["normalized_action"][1][2] = float("nan")
    elif corruption == "inactive":
        response["action"][0][10] = 1.0
    elif corruption == "mask":
        response["action_dim_mask"][10] = True
    else:
        response["future_label"] = "forbidden"
    with pytest.raises(ValueError):
        ledger.accept(response, after_wait=ledger.current)
    assert ledger.phase == "failed"


@pytest.mark.parametrize("changed", ["state_sha256", "sim_time_s", "step"])
def test_physics_or_state_change_while_waiting_invalidates_response(registration, changed):
    config, image = registration
    ledger = CausalStreamLedger(**config)
    response = response_for(issue(ledger, image))
    after_wait = ledger.current
    after_wait[changed] = {"state_sha256": "a" * 64, "sim_time_s": 4.05, "step": 81}[changed]
    with pytest.raises(ValueError):
        ledger.accept(response, after_wait=after_wait)
    assert ledger.phase == "failed"


def test_no_next_request_before_eight_completed_periods(registration):
    config, image = registration
    ledger = CausalStreamLedger(**config)
    ledger.accept(response_for(issue(ledger, image)), after_wait=ledger.current)
    with pytest.raises(ValueError):
        issue(ledger, image)
    assert ledger.phase == "failed"


def test_duplicate_response_is_never_accepted(registration):
    config, image = registration
    ledger = CausalStreamLedger(**config)
    response = response_for(issue(ledger, image))
    ledger.accept(response, after_wait=ledger.current)
    with pytest.raises(ValueError):
        ledger.accept(response, after_wait=ledger.current)
    assert ledger.phase == "failed"


def test_wrong_period_receipt_is_not_committed(registration):
    config, image = registration
    ledger = CausalStreamLedger(**config)
    ledger.accept(response_for(issue(ledger, image)), after_wait=ledger.current)
    before = ledger.current
    after = dict(before, step=81, sim_time_s=4.05)
    with pytest.raises(ValueError):
        ledger.complete_period(before=before, after=after, inner_samples=24)
    assert ledger.completed_periods == 0
    assert ledger.phase == "failed"


def test_rendering_must_preserve_current_snapshot(registration):
    config, image = registration
    ledger = CausalStreamLedger(**config)
    before = ledger.current
    with pytest.raises(ValueError):
        ledger.issue(before_render=before, after_render=dict(before, step=81), image=image)
    assert ledger.issued_requests == 0
    assert ledger.phase == "failed"


def test_returned_snapshots_and_requests_cannot_mutate_the_pending_identity(registration):
    config, image = registration
    ledger = CausalStreamLedger(**config)
    snapshot = ledger.current
    snapshot["step"] = 81
    request = issue(ledger, image)
    altered = copy.deepcopy(request)
    request["image"]["sha256"] = "a" * 64
    assert ledger.current["step"] == 80
    ledger.accept(response_for(altered), after_wait=ledger.current)
    assert ledger.phase == "executing"


def test_service_rejects_duplicated_and_future_requests(registration):
    config, image = registration
    request = issue(CausalStreamLedger(**config), image)
    service = CausalModelAdmission(**config)
    service.admit(request)
    with pytest.raises(ValueError):
        service.admit(request)
    assert service.phase == "failed"
    future = dict(request, request_id=1, step=88, sim_time_s=4.4, image={"path": "not-read.png", "sha256": "a" * 64})
    service = CausalModelAdmission(**config)
    with pytest.raises(ValueError, match="registered sequence"):
        service.admit(future)
    assert service.phase == "failed"


def test_recorded_response_completion_uses_the_historical_sampling_count(registration):
    config, image = registration
    request = issue(CausalStreamLedger(**config), image)
    service = CausalModelAdmission(**config)
    service.admit(request)
    service.complete_recorded(request_id=0, historical_denoise_calls=35)
    assert service.phase == "ready" and service.admitted_requests == 1
    other = CausalModelAdmission(**config)
    other.admit(request)
    with pytest.raises(ValueError):
        other.complete_recorded(request_id=0, historical_denoise_calls=34)
    assert other.phase == "failed"


def test_service_rejects_unregistered_fields_and_changed_image(registration, tmp_path):
    config, image = registration
    request = issue(CausalStreamLedger(**config), image)
    with pytest.raises(ValueError):
        model_inputs(dict(request, future_image="secret"), image_root=config["image_root"])
    (tmp_path / "frame.png").write_bytes(b"modified after request")
    with pytest.raises(ValueError):
        model_inputs(request, image_root=config["image_root"])


@pytest.mark.parametrize("path", ["../outside.png", "/etc/passwd", "frame-link.png"])
def test_image_paths_cannot_escape_or_follow_symlinks(registration, tmp_path, path):
    config, image = registration
    (tmp_path / "frame-link.png").symlink_to(tmp_path / "frame.png")
    image["path"] = path
    ledger = CausalStreamLedger(**config)
    with pytest.raises((ValueError, OSError)):
        issue(ledger, image)
    assert ledger.phase == "failed"


def test_actual_child_process_admits_request_and_returns_one_mock_response(registration):
    config, image = registration
    ledger = CausalStreamLedger(**config)
    parent, child = socket.socketpair()
    channel = FramedJSONChannel(parent, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    code = """
import json, socket, sys
from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission
channel = FramedJSONChannel(socket.socket(fileno=int(sys.argv[1])), max_bytes=2*1024*1024, timeout_seconds=10)
service = CausalModelAdmission(**json.loads(sys.argv[2]))
request = channel.receive()
inputs = service.admit(request)
assert set(inputs) == {'image_bytes', 'task'}
assert inputs['image_bytes'] == b'current synthetic camera frame'
identity = {key: request[key] for key in ('version','session','request_id','step','sim_time_s','state_sha256','seed','model_sha256')}
response = dict(identity, image_sha256=request['image']['sha256'], denoise_calls=35,
                action_dim_mask=[True]*10+[False]*118, action=[[0.0]*128 for _ in range(40)],
                normalized_action=[[0.0]*128 for _ in range(40)])
service.complete(request_id=request['request_id'], denoise_calls=35)
channel.send(response)
channel.close()
"""
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(child.fileno()), json.dumps(config)],
            pass_fds=(child.fileno(),),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        )
        child.close()
        channel.send(issue(ledger, image))
        ledger.accept(channel.receive(), after_wait=ledger.current)
        assert process.wait(timeout=10) == 0
        assert ledger.phase == "executing"
    finally:
        channel.close()
        child.close()
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
