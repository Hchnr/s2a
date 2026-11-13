"""Exercise registered multiseed sampling through the CPU-only model loop."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop
from scripts.wam.tracking.preview import adaptive_live_requests as legacy
from scripts.wam.tracking.preview import adaptive_multiseed_contract as contract
from scripts.wam.tracking.preview import adaptive_multiseed_requests as multi
from tests.research.wam.test_adaptive_live_requests import setup as legacy_setup
from tests.research.wam.test_adaptive_multiseed_contract import template
from tests.research.wam.test_reset_live_request import Overrides, Torch
from tests.research.wam.test_reset_multitask_live_requests import Channel


def setup(tmp_path, task, seed):
    items = legacy_setup(tmp_path)
    registration, _image_root, _directory, requests, _counter, _evidence, _result, service = items
    registration.update(task=task, seed=seed)
    for request in requests:
        request.update(task=task, seed=seed)
    service.task = task
    return items


def call(items, requests, tmp_path, *, torch=Torch, builder=None, gate=None, corrupt_identity=False):
    registration, image_root, directory, _requests, counter, evidence, result, service = items
    checkpoint = {"run": str(tmp_path / "run"), "iteration": "iter_8400"}
    original_execute = service.execute

    def execute(sample, batch):
        response = original_execute(sample, batch).model_dump(mode="json")
        response["args"] = copy.deepcopy(service.requests[-1])
        response["outputs"][0]["content"]["action128"].update(
            seed=42 if corrupt_identity else registration["seed"],
            action_fps=20,
            checkpoint_path=str(tmp_path / "run/checkpoints/iter_8400/model"),
            action_units="raw_physical_or_source_units",
            action_dim_mask=[True] * 10 + [False] * 118,
        )
        return SimpleNamespace(model_dump=lambda mode: response)

    service.execute = execute
    channel = Channel(requests)
    kwargs = dict(
        registration=registration,
        channel=channel,
        image_root=image_root,
        directory=directory,
        template=template(),
        checkpoint=checkpoint,
        service=service,
        torch=torch,
        counter=counter,
        overrides=Overrides,
        domain_reader=lambda *_args: SimpleNamespace(is_libero=True),
        mask_evidence=evidence,
        result=result,
        response_gate=gate,
    )
    if builder is not None:
        kwargs["request_builder"] = builder
    multi.serve_adaptive_requests(**kwargs)
    return channel


@pytest.mark.parametrize("task", contract.TASKS)
@pytest.mark.parametrize("seed", [42, 43])
def test_every_current_request_resets_and_reports_registered_seed(tmp_path, monkeypatch, task, seed):
    items = setup(tmp_path, task, seed)
    calls = []
    monkeypatch.setattr(multi, "random", SimpleNamespace(seed=lambda value: calls.append(("python", value))))
    monkeypatch.setattr(
        multi, "np", SimpleNamespace(random=SimpleNamespace(seed=lambda value: calls.append(("numpy", value))))
    )
    torch = SimpleNamespace(
        manual_seed=lambda value: calls.append(("torch", value)),
        cuda=SimpleNamespace(manual_seed_all=lambda value: calls.append(("cuda", value))),
        inference_mode=nullcontext,
    )
    channel = call(items, items[3], tmp_path, torch=torch)
    result, service = items[6], items[7]
    assert calls == [(name, seed) for _ in range(3) for name in ("python", "numpy", "torch", "cuda")]
    assert result["sampling_seed"] == seed and result["model_sequence_complete"]
    assert result["model_requests_attempted"] == result["model_requests_completed"] == result["responses_sent"] == 3
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 105
    assert [response["step"] for response in channel.sent] == [80, 83, 87]
    assert all(response["seed"] == seed for response in channel.sent)
    assert all(request["seed"] == seed and request["prompt"] == task for request in service.requests)
    assert all(row["reference_sent"] for row in result["responses"])


def test_seed_42_matches_legacy_requests_and_envelopes(tmp_path):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old = setup(old_dir, contract.TASKS[1], 42)
    new = setup(new_dir, contract.TASKS[1], 42)
    new_channel = call(new, new[3], new_dir)
    old_channel = Channel(old[3])
    legacy.serve_adaptive_requests(
        registration=old[0],
        channel=old_channel,
        image_root=old[1],
        directory=old[2],
        template=template(),
        checkpoint={},
        service=old[7],
        torch=Torch,
        counter=old[4],
        overrides=Overrides,
        domain_reader=lambda *_args: SimpleNamespace(is_libero=True),
        mask_evidence=old[5],
        result=old[6],
        response_validator=lambda *_args, **_kwargs: None,
    )
    assert new_channel.sent == old_channel.sent
    for actual, previous in zip(new[7].requests, old[7].requests, strict=True):
        actual, previous = copy.deepcopy(actual), copy.deepcopy(previous)
        actual["vision_path"] = previous["vision_path"] = "<same current image>"
        assert actual == previous
    assert old[6]["denoise_calls_completed"] == new[6]["denoise_calls_completed"] == 105


@pytest.mark.parametrize("seed", [True, 44, "43", None])
def test_bad_seed_fails_before_model_use(tmp_path, seed):
    items = setup(tmp_path, contract.TASKS[0], 42)
    items[0]["seed"] = seed
    with pytest.raises(ValueError, match="42 or 43"):
        call(items, items[3], tmp_path)
    assert not items[7].requests and not items[4].calls


@pytest.mark.parametrize("change", ["seed", "task", "image"])
def test_tampered_actual_request_is_rejected_before_model_preparation(tmp_path, change):
    items = setup(tmp_path, contract.TASKS[0], 43)

    def tamper(template, image, task, seed):
        request = contract.make_task_request(template, image, task, seed)
        request[{"seed": "seed", "task": "prompt", "image": "vision_path"}[change]] = {
            "seed": 42,
            "task": contract.TASKS[1],
            "image": str(tmp_path / "other.png"),
        }[change]
        return request

    with pytest.raises(ValueError, match="actual model request"):
        call(items, items[3], tmp_path, builder=tamper)
    assert not items[7].requests and not items[4].calls and items[6]["responses_sent"] == 0


def test_mixed_request_seed_stops_before_next_model_use(tmp_path):
    items = setup(tmp_path, contract.TASKS[0], 43)
    requests = copy.deepcopy(items[3])
    requests[1]["seed"] = 42
    with pytest.raises(ValueError, match="registered adaptive sequence"):
        call(items, requests, tmp_path)
    assert len(items[7].requests) == items[6]["responses_sent"] == 1
    assert items[6]["denoise_calls_completed"] == 35


def test_mismatched_actual_response_seed_is_not_dispatched(tmp_path):
    items = setup(tmp_path, contract.TASKS[0], 43)
    with pytest.raises(ValueError, match="checkpoint, seed, task or mask"):
        call(items, items[3], tmp_path, corrupt_identity=True)
    assert items[6]["model_requests_completed"] == 1 and items[6]["denoise_calls_completed"] == 35
    assert items[6]["responses_sent"] == 0 and not items[6]["responses"][0]["reference_sent"]


def test_early_stop_and_partial_failure_keep_actual_cost(tmp_path):
    stop_dir, fail_dir = tmp_path / "stop", tmp_path / "failure"
    stop_dir.mkdir()
    fail_dir.mkdir()
    stopped = setup(stop_dir, contract.TASKS[0], 43)
    stop = make_stop(session=stopped[0]["session"], next_step=86)
    channel = call(stopped, [*stopped[3][:2], stop], stop_dir)
    assert len(channel.sent) == stopped[6]["responses_sent"] == 2
    assert stopped[6]["model_sequence_stop"] == stop and stopped[6]["denoise_calls_completed"] == 70
    failed = setup(fail_dir, contract.TASKS[0], 43)

    def fail(_sample, _batch):
        failed[4].calls.extend({"completed": True} for _ in range(3))
        raise RuntimeError("synthetic partial denoise")

    failed[7].execute = fail
    with pytest.raises(RuntimeError, match="partial denoise"):
        call(failed, failed[3], fail_dir)
    assert failed[6]["model_requests_attempted"] == 1 and failed[6]["model_requests_completed"] == 0
    assert failed[6]["denoise_calls_completed"] == 3 and failed[6]["responses_sent"] == 0


def test_changed_input_after_inference_is_not_dispatched(tmp_path):
    items = setup(tmp_path, contract.TASKS[0], 43)
    image = items[1] / items[3][0]["image"]["path"]
    original = items[7].execute

    def change(sample, batch):
        response = original(sample, batch)
        image.write_bytes(b"changed after admission")
        return response

    items[7].execute = change
    with pytest.raises(ValueError):
        call(items, items[3], tmp_path)
    assert items[6]["model_requests_completed"] == 1 and items[6]["denoise_calls_completed"] == 35
    assert items[6]["responses_sent"] == 0
