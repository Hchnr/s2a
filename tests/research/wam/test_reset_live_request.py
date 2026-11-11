"""A fake service must bind one new request to the actual current endpoint."""

import hashlib
import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from s2a.infrastructure.repository_paths import repository_root
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from scripts.wam.tracking.preview.reset_live_request import serve_one


class Channel:
    def __init__(self, request):
        self.request, self.sent = request, []

    def receive(self):
        return self.request

    def send(self, value):
        self.sent.append(value)


class Counter:
    def __init__(self):
        self.calls = []

    @contextmanager
    def scope(self, _name, _count):
        yield


class Torch:
    cuda = SimpleNamespace(manual_seed_all=lambda _seed: None)
    manual_seed = staticmethod(lambda _seed: None)
    inference_mode = staticmethod(nullcontext)


class Overrides:
    model_validate = staticmethod(lambda value: value)


class Service:
    def __init__(self, counter, evidence, *, bad_caption=False, partial=False):
        self.counter, self.evidence = counter, evidence
        self.bad_caption, self.partial = bad_caption, partial
        self.prepare_calls = self.execute_calls = 0

    def prepare(self, _request, _directory):
        self.prepare_calls += 1
        caption = {"task_title": "task", "command": "task", "task_plan": ["future"] if self.bad_caption else []}
        return {}, {"ai_caption": [json.dumps(caption)]}

    def execute(self, _sample, _batch):
        self.execute_calls += 1
        self.counter.calls.extend({"completed": True} for _ in range(2 if self.partial else 35))
        if self.partial:
            raise RuntimeError("partial denoise")
        self.evidence.append(dict(stage="initial_noise", inactive_rms_before=0.1, inactive_rms_after=0.0))
        array = [[0.0] * 128 for _ in range(40)]
        return SimpleNamespace(
            model_dump=lambda mode: {
                "status": "success",
                "outputs": [{"content": {"action": array, "action128": {"normalized_action": array}}}],
            }
        )


def setup(tmp_path):
    root = repository_root()
    capture, directory = tmp_path / "captures", tmp_path / "model"
    capture.mkdir()
    directory.mkdir()
    data = b"the currently observed image"
    (capture / "current.png").write_bytes(data)
    registration = dict(
        session="reset-test",
        initial=dict(step=8, sim_time_s=0.65, state_sha256="a" * 64),
        image_root=str(capture.relative_to(root)),
        task="task",
        seed=42,
        model_sha256="b" * 64,
        max_requests=1,
        denoise_steps=35,
    )
    ledger = CausalStreamLedger(**registration)
    request = ledger.issue(
        before_render=ledger.current,
        after_render=ledger.current,
        image=dict(path="current.png", sha256=hashlib.sha256(data).hexdigest()),
    )
    return registration, capture, directory, request


def execute(tmp_path, *, failure=None):
    registration, capture, directory, request = setup(tmp_path)
    if failure == "wrong_state":
        request["state_sha256"] = "c" * 64
    if failure == "changed_image":
        (capture / "current.png").write_bytes(b"not the admitted image")
    channel, counter, evidence, result = Channel(request), Counter(), [], {}
    service = Service(counter, evidence, bad_caption=failure == "caption", partial=failure == "partial")
    args = dict(
        registration=registration,
        channel=channel,
        image_root=capture,
        directory=directory,
        template={},
        checkpoint={},
        service=service,
        torch=Torch,
        counter=counter,
        overrides=Overrides,
        domain_reader=lambda *_args: SimpleNamespace(is_libero=True),
        mask_evidence=evidence,
        result=result,
        request_builder=lambda _template, image: {"name": "synthetic", "vision_path": image["path"]},
        response_validator=lambda *_args, **_kwargs: None,
    )
    return args, service, channel, result


def test_one_request_uses_only_current_image_and_records_actual_response(tmp_path):
    args, service, channel, result = execute(tmp_path)
    serve_one(**args)
    assert service.prepare_calls == service.execute_calls == result["responses_sent"] == 1
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 35
    assert channel.sent[0]["state_sha256"] == args["registration"]["initial"]["state_sha256"]
    assert result["responses"][0]["reference_sent"] is True
    assert args["registration"]["task"] == "task"


@pytest.mark.parametrize("failure", ["wrong_state", "changed_image", "caption", "partial"])
def test_invalid_input_or_partial_model_output_never_dispatches_reference(tmp_path, failure):
    args, service, channel, result = execute(tmp_path, failure=failure)
    with pytest.raises((ValueError, RuntimeError)):
        serve_one(**args)
    assert not channel.sent and result["responses_sent"] == 0
    if failure in ("wrong_state", "changed_image"):
        assert service.prepare_calls == 0 and result["model_requests_attempted"] == 0
    if failure == "partial":
        assert result["denoise_calls_attempted"] == 2 and result["model_requests_completed"] == 0
