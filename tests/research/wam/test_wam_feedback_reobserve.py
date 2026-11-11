import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts.wam.tracking.preview import wam_feedback_reobserve_20261111 as reobserve


@pytest.fixture(scope="module")
def inputs():
    config = json.loads(reobserve.CONFIG.read_text())
    reobserve.validate_config(config)
    captures = reobserve.capture_inputs(config)
    previous = reobserve.read_ref(config, "previous_response")
    return config, captures, previous


def test_current_images_bind_the_recorded_feedback_endpoints(inputs):
    config, captures, _previous = inputs
    assert [item["frame_index"] for item in captures] == [80, 88]
    assert captures[0]["canvas_png"]["sha256"] == config["images"][0]["sha256"]
    assert captures[1]["canvas_png"]["sha256"] == config["images"][1]["sha256"]
    assert all(item["render_before"] == item["render_after"] for item in captures)


def test_second_request_is_only_admitted_after_exact_first_response(inputs, tmp_path):
    _config, _captures, previous = inputs
    gate = reobserve.FirstResponseGate(previous)
    with pytest.raises(ValueError, match="second request requires"):
        gate.before_request({"step": 88})
    gate.before_request({"step": 80})
    result = gate({"step": 80}, copy.deepcopy(previous), tmp_path)
    assert result["passed"] is True
    assert all(item["elementwise_equal"] for item in result["comparison"].values())
    assert json.loads((tmp_path / "causal_match.json").read_text())["passed"] is True
    gate.before_request({"step": 88})
    assert gate({"step": 88}, previous, tmp_path)["first_response_exact"] is True
    with pytest.raises(ValueError, match="second request requires"):
        gate.before_request({"step": 88})


def test_changed_first_response_is_recorded_and_blocks_second_request(inputs, tmp_path):
    _config, _captures, previous = inputs
    changed = copy.deepcopy(previous)
    changed["outputs"][0]["content"]["action"][0][0] += 0.001
    gate = reobserve.FirstResponseGate(previous)
    gate.before_request({"step": 80})
    with pytest.raises(ValueError, match="differs from the previously executed"):
        gate({"step": 80}, changed, tmp_path)
    assert json.loads((tmp_path / "causal_match.json").read_text())["passed"] is False
    with pytest.raises(ValueError, match="second request requires"):
        gate.before_request({"step": 88})


def test_changed_current_image_is_rejected_before_model_loading(inputs):
    config = copy.deepcopy(inputs[0])
    config["images"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Current image"):
        reobserve.capture_inputs(config)


@pytest.mark.parametrize("first_matches", [True, False])
def test_complete_request_loop_enforces_gate_with_simulated_service(inputs, tmp_path, first_matches):
    config, _captures, previous = inputs
    _prepared, _bridge, _completed, template, _response = reobserve.engine.previous_inputs()
    first = copy.deepcopy(previous)
    if not first_matches:
        first["outputs"][0]["content"]["action"][0][0] += 0.001

    class FakeService:
        def __init__(self):
            self.calls = []

        def prepare(self, request, _directory):
            caption = {"task_title": reobserve.TASK, "command": reobserve.TASK, "task_plan": []}
            return request, {"ai_caption": [json.dumps(caption)]}

        def execute(self, sample, _batch):
            self.calls.append(sample["name"])
            response = first if len(self.calls) == 1 else previous
            return SimpleNamespace(model_dump=lambda **_kwargs: copy.deepcopy(response))

    class FakeTorch:
        cuda = SimpleNamespace(manual_seed_all=lambda _seed: None)
        manual_seed = staticmethod(lambda _seed: None)
        inference_mode = staticmethod(nullcontext)

    class FakeCounter:
        @staticmethod
        def scope(_label, count):
            assert count == 35
            return nullcontext()

    class FakeOverrides:
        model_validate = staticmethod(lambda request: request)

    service = FakeService()
    context = reobserve.context(config)
    result, responses = {"model_requests": 0}, []
    args = (
        config,
        context,
        service,
        FakeTorch,
        FakeCounter,
        FakeOverrides,
        lambda _service, _sample, _batch: SimpleNamespace(is_libero=True),
        tmp_path,
        template,
        result,
        responses,
    )
    if first_matches:
        reobserve.engine.execute_requests(*args)
        assert result["model_requests"] == 2 and len(responses) == len(service.calls) == 2
        assert responses[0]["guard"]["passed"] is True
        assert responses[1]["guard"]["first_response_exact"] is True
    else:
        with pytest.raises(ValueError, match="differs from the previously executed"):
            reobserve.engine.execute_requests(*args)
        assert result["model_requests"] == 1 and len(responses) == len(service.calls) == 1
        assert not (tmp_path / "step88").exists()
