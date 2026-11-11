"""Exercise the real request loop with a fake service and no GPU."""

import copy
import json
import shutil
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.wam.tracking.preview import causal_stream_live_admission as admission
from scripts.wam.tracking.preview import live_stream_requests as loop


class Channel:
    def __init__(self, requests):
        self.requests = requests
        self.responses = []

    def receive(self):
        return self.requests.pop(0)

    def send(self, response):
        self.responses.append(response)


class Counter:
    def __init__(self):
        self.calls = []

    @contextmanager
    def scope(self, _name, count):
        before = len(self.calls)
        yield
        assert len(self.calls) - before == count


class FakeOverrides:
    @staticmethod
    def model_validate(value):
        return value


class FakeTorch:
    cuda = SimpleNamespace(manual_seed_all=lambda _seed: None)

    @staticmethod
    def manual_seed(_seed):
        pass

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeService:
    def __init__(self, inputs, counter, evidence, *, failure=None, captured_image=None):
        self.inputs, self.counter, self.evidence = inputs, counter, evidence
        self.failure, self.captured_image = failure, captured_image
        self.request = None
        self.prepare_calls = 0

    def prepare(self, request, _directory):
        self.prepare_calls += 1
        self.request = request
        task = self.inputs["live_config"]["task"]
        caption = {"task_title": task, "command": task, "task_plan": []}
        if self.failure == "caption":
            caption["task_plan"] = ["future label"]
        return {}, {"ai_caption": [json.dumps(caption)]}

    def execute(self, _sample, _batch):
        count = 3 if self.failure == "partial_denoise" else 35
        self.counter.calls.extend({"completed": True} for _ in range(count))
        if self.failure == "partial_denoise":
            raise RuntimeError("synthetic denoise failure")
        if self.failure != "mask":
            self.evidence.append({"stage": "initial_noise", "inactive_rms_before": 0.1, "inactive_rms_after": 0.0})
        index = 0 if "t80_" in self.request["name"] else 1
        ref = self.inputs["old"]["refs"][("response80", "response88")[index]]
        response = json.loads(Path(ref["path"]).read_text())
        response["args"].update(copy.deepcopy(self.request))
        if self.failure == "reference":
            response["outputs"][0]["content"]["action"][0][0] += 0.01
        if self.failure == "response_identity":
            response["args"]["vision_path"] = "stale.png"
        if self.failure == "changed_image":
            self.captured_image.write_bytes(b"changed during inference")
        return SimpleNamespace(model_dump=lambda mode: response)


def setup_loop(tmp_path, *, failure=None):
    inputs = admission.load_inputs()
    image_root, directory = tmp_path / "captures", tmp_path / "model"
    image_root.mkdir()
    directory.mkdir()
    for image in inputs["old"]["images"]:
        target = image_root / f"anchor{image['frame_index']:04d}"
        target.mkdir()
        shutil.copyfile(image["canvas_png"]["path"], target / "current.png")
    ledger = CausalStreamLedger(**physical.model_config(inputs, image_root))
    current = ledger.current
    first = ledger.issue(
        before_render=current,
        after_render=current,
        image={"path": "anchor0080/current.png", "sha256": inputs["records"][0]["image_sha256"]},
    )
    second = copy.deepcopy(first)
    record = inputs["records"][1]
    second.update(request_id=1, step=88, sim_time_s=record["sim_time_s"], state_sha256=record["state_sha256"])
    second["image"] = {"path": "anchor0088/current.png", "sha256": record["image_sha256"]}
    if failure == "stale_request":
        first["state_sha256"] = "a" * 64
    channel, counter, evidence, result = Channel([first, second]), Counter(), [], {}
    service = FakeService(
        inputs, counter, evidence, failure=failure, captured_image=image_root / "anchor0080/current.png"
    )
    return inputs, image_root, directory, channel, counter, evidence, result, service


def run_loop(items):
    inputs, image_root, directory, channel, counter, evidence, result, service = items
    loop.execute_live_requests(
        inputs,
        channel=channel,
        image_root=image_root,
        directory=directory,
        service=service,
        torch=FakeTorch,
        counter=counter,
        overrides=FakeOverrides,
        domain_reader=lambda *_args: SimpleNamespace(is_libero=True),
        mask_evidence=evidence,
        result=result,
    )


def test_two_actual_request_loop_iterations_use_only_current_registered_images(tmp_path):
    items = setup_loop(tmp_path)
    run_loop(items)
    inputs, _root, directory, channel, _counter, evidence, result, service = items
    assert result["local_requests_received"] == result["model_requests_attempted"] == 2
    assert result["model_requests_completed"] == result["responses_sent"] == 2
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 70
    assert len(evidence) == service.prepare_calls == len(channel.responses) == 2
    assert all(row["reference_sent"] for row in result["responses"])
    for index, response in enumerate(channel.responses):
        assert response["step"] == inputs["records"][index]["step"]
        np.testing.assert_array_equal(response["action"], inputs["records"][index]["action"])
        request = json.loads((directory / f"step{response['step']}" / "request.json").read_text())
        assert request["vision_path"].startswith(str(directory))


@pytest.mark.parametrize("failure", ["reference", "response_identity", "changed_image", "mask"])
def test_invalid_completed_output_is_saved_and_never_sent(tmp_path, failure):
    items = setup_loop(tmp_path, failure=failure)
    with pytest.raises(ValueError):
        run_loop(items)
    _inputs, _root, _directory, channel, _counter, _evidence, result, service = items
    assert service.prepare_calls == result["model_requests_attempted"] == result["model_requests_completed"] == 1
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 35
    assert result["responses_sent"] == len(channel.responses) == 0
    assert len(result["responses"]) == 1 and not result["responses"][0]["reference_sent"]
    assert Path(result["responses"][0]["actual_response_ref"]["path"]).is_file()


def test_partial_denoise_failure_preserves_actual_attempt_counts(tmp_path):
    items = setup_loop(tmp_path, failure="partial_denoise")
    with pytest.raises(RuntimeError, match="denoise failure"):
        run_loop(items)
    result = items[6]
    assert result["model_requests_attempted"] == 1 and result["model_requests_completed"] == 0
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 3
    assert result["responses_sent"] == 0 and result["local_requests_received"] == 1


@pytest.mark.parametrize("failure", ["stale_request", "caption"])
def test_bad_current_binding_or_caption_stops_before_inference(tmp_path, failure):
    items = setup_loop(tmp_path, failure=failure)
    with pytest.raises(ValueError):
        run_loop(items)
    result = items[6]
    assert result["local_requests_received"] == 1 and result["model_requests_attempted"] == 0
    assert result["denoise_calls_attempted"] == result["responses_sent"] == 0
