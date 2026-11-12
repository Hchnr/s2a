"""Exercise the real adaptive model loop using a synthetic CPU-only service."""

import copy
import hashlib
from types import SimpleNamespace

import pytest

from s2a.infrastructure.repository_paths import repository_root
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop
from scripts.wam.tracking.preview import adaptive_live_requests as adaptive
from tests.research.native_runtime.test_streaming_continuation import response_for
from tests.research.wam.test_reset_live_request import Counter, Overrides, Torch
from tests.research.wam.test_reset_multitask_live_requests import Channel, Service


def setup(tmp_path):
    root = repository_root()
    image_root, directory = tmp_path / "captures", tmp_path / "model"
    image_root.mkdir()
    directory.mkdir()
    registration = {
        "session": "adaptive-model-test",
        "initial": {"step": 80, "sim_time_s": 4.0, "state_sha256": "a" * 64},
        "image_root": str(image_root.relative_to(root)),
        "task": "put the bowl on the plate",
        "seed": 42,
        "model_sha256": "b" * 64,
        "max_requests": 3,
        "max_periods": 12,
        "denoise_steps": 35,
    }
    ledger, requests = AdaptiveStreamLedger(**registration), []
    for index, count in enumerate((3, 4, 1)):
        data = f"current image {index}".encode()
        name = f"frame{index}.png"
        (image_root / name).write_bytes(data)
        current = ledger.current
        request = ledger.issue(
            before_render=current,
            after_render=current,
            image={"path": name, "sha256": hashlib.sha256(data).hexdigest()},
        )
        requests.append(request)
        ledger.accept(response_for(request), after_wait=current)
        for _ in range(count):
            before = ledger.current
            after = {
                "step": before["step"] + 1,
                "sim_time_s": before["sim_time_s"] + 0.05,
                "state_sha256": hashlib.sha256(f"state{before['step'] + 1}".encode()).hexdigest(),
            }
            ledger.begin_period(before=before)
            ledger.complete_period(before=before, after=after, inner_samples=25)
        if index < 2:
            assert ledger.refresh_early(trigger={"replan": True, "current_step": ledger.current["step"], "reason": "q"})
    counter, evidence, result = Counter(), [], {}
    service = Service(registration["task"], counter, evidence)
    return registration, image_root, directory, requests, counter, evidence, result, service


def call(items, requests, *, response_gate=None):
    registration, image_root, directory, _requests, counter, evidence, result, service = items
    channel = Channel(requests)
    adaptive.serve_adaptive_requests(
        registration=registration,
        channel=channel,
        image_root=image_root,
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
        response_gate=response_gate,
        request_builder=lambda _template, image, task: {
            "name": f"synthetic_{image['step']}",
            "vision_path": image["path"],
            "task": task,
        },
        response_validator=lambda *_args, **_kwargs: None,
    )
    return channel


def test_variable_cadence_serves_each_current_image_with_fixed_sampling(tmp_path):
    items = setup(tmp_path)
    channel = call(items, items[3])
    result, service = items[6], items[7]
    assert result["model_sequence_complete"] and result["model_sequence_stop"] is None
    assert result["model_requests_attempted"] == result["model_requests_completed"] == result["responses_sent"] == 3
    assert result["response_send_attempts"] == 3
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 105
    assert [response["step"] for response in channel.sent] == [80, 83, 87]
    assert all(request["task"] == items[0]["task"] for request in service.requests)
    assert all(row["reference_sent"] for row in result["responses"])


def test_early_stop_after_partial_segment_keeps_only_actual_model_work(tmp_path):
    items = setup(tmp_path)
    stop = make_stop(session=items[0]["session"], next_step=86)
    channel = call(items, [*items[3][:2], stop])
    result = items[6]
    assert len(channel.sent) == result["responses_sent"] == 2
    assert not result["model_sequence_complete"] and result["model_sequence_stop"] == stop
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 70


@pytest.mark.parametrize("change", ["first_state", "duplicate_step", "future_step", "wrong_seed"])
def test_invalid_sequence_stops_before_an_unregistered_model_request(tmp_path, change):
    items = setup(tmp_path)
    requests = copy.deepcopy(items[3])
    if change == "first_state":
        requests[0]["state_sha256"] = "f" * 64
        expected_calls = 0
    else:
        expected_calls = 1
        if change == "duplicate_step":
            requests[1].update(step=80, sim_time_s=4.0)
        elif change == "future_step":
            requests[1].update(step=89, sim_time_s=4.45)
        else:
            requests[1]["seed"] = 43
    with pytest.raises(ValueError):
        call(items, requests)
    assert items[6]["model_requests_attempted"] == items[6]["responses_sent"] == expected_calls
    assert items[6]["denoise_calls_completed"] == expected_calls * 35


def test_response_gate_rejection_keeps_inference_evidence_and_sends_nothing(tmp_path):
    items = setup(tmp_path)
    with pytest.raises(ValueError, match="response gate"):
        call(items, items[3], response_gate=lambda _request, _response: False)
    result = items[6]
    assert result["model_requests_attempted"] == result["model_requests_completed"] == 1
    assert result["denoise_calls_completed"] == 35 and result["responses_sent"] == 0
    assert not result["responses"][0]["reference_sent"]


def test_changed_input_image_after_inference_is_never_sent(tmp_path):
    items = setup(tmp_path)
    service, image = items[7], items[1] / items[3][0]["image"]["path"]
    original = service.execute

    def change_image(sample, batch):
        response = original(sample, batch)
        image.write_bytes(b"changed during inference")
        return response

    service.execute = change_image
    with pytest.raises(ValueError):
        call(items, items[3])
    assert items[6]["model_requests_completed"] == 1 and items[6]["denoise_calls_completed"] == 35
    assert items[6]["responses_sent"] == 0 and not items[6]["responses"][0]["reference_sent"]


def test_partial_denoise_failure_preserves_actual_cost(tmp_path):
    items = setup(tmp_path)

    def fail(_sample, _batch):
        items[4].calls.extend({"completed": True} for _ in range(3))
        raise RuntimeError("synthetic denoise failure")

    items[7].execute = fail
    with pytest.raises(RuntimeError, match="synthetic denoise failure"):
        call(items, items[3])
    result = items[6]
    assert result["model_requests_attempted"] == 1 and result["model_requests_completed"] == 0
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 3
    assert result["responses_sent"] == 0


@pytest.mark.parametrize("change", ["task", "seed", "denoise", "absolute_budget", "dirty_counter"])
def test_bad_registration_fails_before_receiving_or_inference(tmp_path, change):
    items = setup(tmp_path)
    registration = items[0]
    if change == "task":
        registration["task"] = "unregistered task"
    elif change == "seed":
        registration["seed"] = 43
    elif change == "denoise":
        registration["denoise_steps"] = 34
    elif change == "absolute_budget":
        registration["max_periods"] = 521
    else:
        items[4].calls.append({"completed": True})
    with pytest.raises(ValueError, match="fresh registered"):
        call(items, items[3])
    assert not items[7].requests
