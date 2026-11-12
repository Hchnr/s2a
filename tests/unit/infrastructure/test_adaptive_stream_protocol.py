"""Exercise variable-cadence admission with synthetic, CPU-only observations."""

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
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import (
    AdaptiveModelAdmission,
    AdaptiveStreamLedger,
    check_adaptive_stop,
)
from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def setup_stream(tmp_path):
    image = b"current synthetic camera frame"
    (tmp_path / "frame.png").write_bytes(image)
    config = {
        "session": "adaptive-cpu-only",
        "initial": {"step": 80, "sim_time_s": 4.0, "state_sha256": digest(b"step80")},
        "image_root": str(tmp_path.relative_to(repository_root())),
        "task": "place the object",
        "seed": 42,
        "model_sha256": digest(b"synthetic model"),
        "max_requests": 4,
        "max_periods": 24,
    }
    return config, {"path": "frame.png", "sha256": digest(image)}


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


def period(ledger, *, inner_samples=25):
    before = ledger.current
    after = {
        "step": before["step"] + 1,
        "sim_time_s": before["sim_time_s"] + 0.05,
        "state_sha256": digest(f"step{before['step'] + 1}".encode()),
    }
    ledger.begin_period(before=before)
    ledger.complete_period(before=before, after=after, inner_samples=inner_samples)


def refresh(ledger, *, step=None):
    return ledger.refresh_early(
        trigger={"replan": True, "current_step": ledger.current["step"] if step is None else step, "reason": "quality"}
    )


def admit_and_accept(ledger, service, image):
    request = issue(ledger, image)
    assert service.admit(request) == {"image_bytes": b"current synthetic camera frame", "task": "place the object"}
    service.complete(request_id=request["request_id"], denoise_calls=35)
    ledger.accept(response_for(request), after_wait=ledger.current)
    return request


def test_variable_segments_preserve_request_identity_and_budgets(setup_stream):
    config, image = setup_stream
    config["max_periods"] = 16
    ledger, service = AdaptiveStreamLedger(**config), AdaptiveModelAdmission(**config)
    steps = []
    for count in (3, 4, 8, 1):
        steps.append(admit_and_accept(ledger, service, image)["step"])
        for _ in range(count):
            period(ledger)
        if count < 8 and ledger.phase == "executing":
            assert refresh(ledger)
    assert steps == [80, 83, 87, 95]
    assert ledger.current["step"] == 96
    assert ledger.completed_periods == 16
    assert ledger.issued_requests == ledger.accepted_requests == service.admitted_requests == 4
    assert ledger.phase == service.phase == "done"
    assert [row["completed_periods"] for row in ledger.early_refreshes] == [3, 4]


@pytest.mark.parametrize("count", range(1, 8))
def test_each_nonempty_partial_segment_can_reobserve(setup_stream, count):
    config, image = setup_stream
    ledger, service = AdaptiveStreamLedger(**config), AdaptiveModelAdmission(**config)
    admit_and_accept(ledger, service, image)
    for _ in range(count):
        period(ledger)
    assert refresh(ledger)
    request = admit_and_accept(ledger, service, image)
    assert request["step"] == 80 + count
    assert request["state_sha256"] == digest(f"step{80 + count}".encode())


def test_full_segment_reobserves_without_early_refresh(setup_stream):
    config, image = setup_stream
    ledger, service = AdaptiveStreamLedger(**config), AdaptiveModelAdmission(**config)
    admit_and_accept(ledger, service, image)
    for _ in range(8):
        period(ledger)
    assert ledger.phase == "ready" and not ledger.early_refreshes
    assert admit_and_accept(ledger, service, image)["step"] == 88


@pytest.mark.parametrize("invalid", ["zero", "pending", "stale", "negative", "complete_without_begin", "short_inner"])
def test_unfinished_or_unverified_period_cannot_be_crossed(setup_stream, invalid):
    config, image = setup_stream
    ledger, service = AdaptiveStreamLedger(**config), AdaptiveModelAdmission(**config)
    admit_and_accept(ledger, service, image)
    if invalid in ("pending", "stale", "negative"):
        period(ledger)
    with pytest.raises(ValueError):
        if invalid == "pending":
            ledger.begin_period(before=ledger.current)
            refresh(ledger)
        elif invalid == "stale":
            refresh(ledger, step=80)
        elif invalid == "negative":
            ledger.refresh_early(trigger={"replan": False, "current_step": 81, "reason": "quality"})
        elif invalid == "complete_without_begin":
            ledger.complete_period(before=ledger.current, after=ledger.current, inner_samples=25)
        elif invalid == "short_inner":
            period(ledger, inner_samples=24)
        else:
            refresh(ledger)
    assert ledger.phase == "failed"
    with pytest.raises(ValueError):
        issue(ledger, image)


def test_period_and_request_budgets_stop_without_another_dispatch(setup_stream):
    config, image = setup_stream
    config.update(max_requests=1, max_periods=5)
    ledger, service = AdaptiveStreamLedger(**config), AdaptiveModelAdmission(**config)
    admit_and_accept(ledger, service, image)
    period(ledger)
    assert refresh(ledger) is False
    assert ledger.phase == "done"
    with pytest.raises(ValueError):
        issue(ledger, image)
    config.update(max_requests=4, max_periods=1)
    ledger, service = AdaptiveStreamLedger(**config), AdaptiveModelAdmission(**config)
    admit_and_accept(ledger, service, image)
    period(ledger)
    assert ledger.phase == "done" and ledger.completed_periods == 1
    with pytest.raises(ValueError):
        ledger.begin_period(before=ledger.current)


@pytest.mark.parametrize("bad", [0, 601, -1, True, 1.5])
def test_invalid_period_budget_rejected(setup_stream, bad):
    config, _image = setup_stream
    config["max_periods"] = bad
    with pytest.raises(ValueError):
        AdaptiveStreamLedger(**config)
    with pytest.raises(ValueError):
        AdaptiveModelAdmission(**config)


@pytest.mark.parametrize("change", ["duplicate", "future", "time", "identity", "id", "task", "limit"])
def test_model_service_rejects_noncausal_or_unregistered_requests(setup_stream, change):
    config, image = setup_stream
    config["max_periods"] = 8
    ledger, service = AdaptiveStreamLedger(**config), AdaptiveModelAdmission(**config)
    request = issue(ledger, image)
    service.admit(request)
    service.complete(request_id=0, denoise_calls=35)
    second = copy.deepcopy(request)
    second.update(request_id=1, step=81, sim_time_s=4.05, state_sha256=digest(b"step81"))
    if change == "duplicate":
        second.update(step=80, sim_time_s=4.0)
    elif change == "future":
        second.update(step=89, sim_time_s=4.45)
    elif change == "time":
        second["sim_time_s"] = 4.1
    elif change == "identity":
        second["model_sha256"] = "a" * 64
    elif change == "id":
        second["request_id"] = 2
    elif change == "task":
        second["task"] = "different task"
    else:
        second.update(step=88, sim_time_s=4.4)
    with pytest.raises(ValueError):
        service.admit(second)
    assert service.phase == "failed"


def test_first_request_binds_registered_state_and_waiting_does_not_advance(setup_stream):
    config, image = setup_stream
    ledger, service = AdaptiveStreamLedger(**config), AdaptiveModelAdmission(**config)
    request = issue(ledger, image)
    altered = dict(request, state_sha256="a" * 64)
    with pytest.raises(ValueError):
        service.admit(altered)
    with pytest.raises(ValueError):
        ledger.accept(response_for(request), after_wait=dict(ledger.current, step=81))
    assert ledger.phase == service.phase == "failed"


def test_stop_frame_is_bounded_by_last_admitted_window_and_budget(setup_stream):
    config, image = setup_stream
    config["max_periods"] = 5
    ledger, service = AdaptiveStreamLedger(**config), AdaptiveModelAdmission(**config)
    assert check_adaptive_stop(service, make_stop(session=config["session"], next_step=80))["next_step"] == 80
    request = issue(ledger, image)
    service.admit(request)
    with pytest.raises(ValueError):
        check_adaptive_stop(service, make_stop(session=config["session"], next_step=80))
    service.complete(request_id=0, denoise_calls=35)
    for step in (80, 83, 85):
        assert check_adaptive_stop(service, make_stop(session=config["session"], next_step=step))["next_step"] == step
    for step in (79, 86):
        with pytest.raises(ValueError):
            check_adaptive_stop(service, make_stop(session=config["session"], next_step=step))


def test_real_child_service_accepts_two_observations_after_three_periods(setup_stream):
    config, image = setup_stream
    ledger = AdaptiveStreamLedger(**config)
    parent, child = socket.socketpair()
    channel = FramedJSONChannel(parent, max_bytes=2 * 1024 * 1024, timeout_seconds=10)
    code = """
import json, socket, sys
from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveModelAdmission
channel = FramedJSONChannel(socket.socket(fileno=int(sys.argv[1])), max_bytes=2*1024*1024, timeout_seconds=10)
service = AdaptiveModelAdmission(**json.loads(sys.argv[2]))
for step in (80, 83):
    request = channel.receive()
    assert request['step'] == step
    assert service.admit(request) == {'image_bytes': b'current synthetic camera frame', 'task': 'place the object'}
    identity = {k: request[k] for k in ('version','session','request_id','step','sim_time_s','state_sha256','seed','model_sha256')}
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
        for _ in range(3):
            period(ledger)
        assert refresh(ledger)
        channel.send(issue(ledger, image))
        ledger.accept(channel.receive(), after_wait=ledger.current)
        assert process.wait(timeout=10) == 0
    finally:
        channel.close()
        child.close()
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
