"""Test seed-43 current-image service through the real causal request loop."""

import copy
import hashlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_seed43_live_admission as admission
from scripts.wam.tracking.preview import reset_live_request as legacy
from scripts.wam.tracking.preview import reset_seed43_live_request as service_loop
from tests.research.wam.test_adaptive_multiseed_contract import template
from tests.research.wam.test_reset_live_request import Channel, Counter, Overrides, Torch
from tests.research.wam.test_reset_multitask_live_requests import Service


def setup(tmp_path):
    output = tmp_path / "trial"
    image_root = output / "producer/simulation/runtime"
    directory = output / "producer/model/model_run"
    image_root.mkdir(parents=True)
    directory.mkdir(parents=True)
    image = b"observed image after eight completed periods"
    (image_root / "current.png").write_bytes(image)
    config = prefix.read({"path": str(admission.CONFIG), "sha256": admission.CONFIG_SHA})
    registration = dict(
        session=admission.SESSION,
        initial=dict(step=8, sim_time_s=0.6500000000000005, state_sha256=config["historical_step8_state_sha256"]),
        image_root=str(image_root.relative_to(prefix.REPO)),
        task=config["task"],
        seed=43,
        model_sha256=config["model_sha256"],
        max_requests=1,
        denoise_steps=35,
    )
    ledger = CausalStreamLedger(**registration)
    request = ledger.issue(
        before_render=ledger.current,
        after_render=ledger.current,
        image=dict(path="current.png", sha256=hashlib.sha256(image).hexdigest()),
    )
    counter, evidence, result = Counter(), [], {}
    service = Service(registration["task"], counter, evidence)
    checkpoint = dict(run=str(output / "weights"), iteration="iter_000008400")
    return output, registration, image_root, directory, request, counter, evidence, result, service, checkpoint


def call(items, request, *, torch=Torch, corrupt_identity=False):
    output, registration, image_root, directory, _request, counter, evidence, result, service, checkpoint = items
    original = service.execute

    def execute(sample, batch):
        response = original(sample, batch).model_dump(mode="json")
        response["args"] = copy.deepcopy(service.requests[-1])
        response["outputs"][0]["content"]["action128"].update(
            seed=42 if corrupt_identity else 43,
            action_fps=20,
            checkpoint_path=str(output / "weights/checkpoints/iter_000008400/model"),
            action_units="raw_physical_or_source_units",
            action_dim_mask=[True] * 10 + [False] * 118,
        )
        return SimpleNamespace(model_dump=lambda mode: response)

    service.execute = execute
    channel = Channel(request)
    service_loop.serve_seed43(
        output=output,
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
    )
    return channel


def test_one_current_image_resets_and_sends_seed43(tmp_path, monkeypatch):
    items = setup(tmp_path)
    calls = []
    monkeypatch.setattr(legacy, "random", SimpleNamespace(seed=lambda value: calls.append(("python", value))))
    monkeypatch.setattr(
        legacy, "np", SimpleNamespace(random=SimpleNamespace(seed=lambda value: calls.append(("numpy", value))))
    )
    torch = SimpleNamespace(
        manual_seed=lambda value: calls.append(("torch", value)),
        cuda=SimpleNamespace(manual_seed_all=lambda value: calls.append(("cuda", value))),
        inference_mode=nullcontext,
    )
    channel = call(items, items[4], torch=torch)
    assert calls == [(name, 43) for name in ("python", "numpy", "torch", "cuda")]
    assert items[8].requests[0]["seed"] == channel.sent[0]["seed"] == 43
    assert items[8].requests[0]["prompt"] == items[1]["task"]
    assert items[7]["model_sequence_complete"] and items[7]["model_requests_completed"] == 1
    assert items[7]["denoise_calls_completed"] == 35 and items[7]["responses_sent"] == 1


@pytest.mark.parametrize(
    "change", ["seed", "boolean_seed", "task", "session", "model", "requests", "denoise", "state", "time", "image_root"]
)
def test_bad_registration_stops_before_receiving_or_loading(tmp_path, change):
    items = setup(tmp_path)
    registration = items[1]
    key, value = {
        "seed": ("seed", 42),
        "boolean_seed": ("seed", True),
        "task": ("task", "different task"),
        "session": ("session", "unregistered-session"),
        "model": ("model_sha256", "0" * 64),
        "requests": ("max_requests", 2),
        "denoise": ("denoise_steps", 34),
        "state": ("initial", dict(registration["initial"], state_sha256="0" * 64)),
        "time": ("initial", dict(registration["initial"], sim_time_s=0.65)),
        "image_root": ("image_root", str(tmp_path.relative_to(prefix.REPO))),
    }[change]
    registration[key] = value
    with pytest.raises(ValueError, match="registered seed-43"):
        call(items, items[4])
    assert items[8].requests == [] and items[5].calls == []


def test_runtime_image_directory_may_be_created_by_the_other_worker(tmp_path):
    output = tmp_path / "trial"
    output.mkdir()
    config = prefix.read({"path": str(admission.CONFIG), "sha256": admission.CONFIG_SHA})
    registration = dict(
        session=admission.SESSION,
        initial=dict(step=8, sim_time_s=0.6500000000000005, state_sha256=config["historical_step8_state_sha256"]),
        image_root=str((output / "producer/simulation/runtime").relative_to(prefix.REPO)),
        task=config["task"],
        seed=43,
        model_sha256=config["model_sha256"],
        max_requests=1,
        denoise_steps=35,
    )
    assert service_loop.validate_registration(registration, output).phase == "ready"
    assert not (output / "producer/simulation/runtime").exists()


def test_early_stop_does_not_invent_inference(tmp_path):
    items = setup(tmp_path)
    stop = make_stop(session=admission.SESSION, next_step=8)
    channel = call(items, stop)
    assert channel.sent == [] and items[8].requests == []
    assert items[7]["model_sequence_stop"] == stop and not items[7]["model_sequence_complete"]
    assert items[7]["denoise_calls_attempted"] == 0


def test_response_seed_mismatch_is_not_dispatched(tmp_path):
    items = setup(tmp_path)
    with pytest.raises(ValueError, match="checkpoint, seed, task or mask"):
        call(items, items[4], corrupt_identity=True)
    assert items[7]["model_requests_completed"] == 1 and items[7]["denoise_calls_completed"] == 35
    assert items[7]["responses_sent"] == 0 and not items[7]["responses"][0]["reference_sent"]


def test_changed_current_image_and_partial_denoise_never_send_reference(tmp_path):
    changed_dir, failure_dir = tmp_path / "changed", tmp_path / "failure"
    changed_dir.mkdir()
    failure_dir.mkdir()
    changed = setup(changed_dir)
    original = changed[8].execute

    def change_image(sample, batch):
        response = original(sample, batch)
        (changed[2] / "current.png").write_bytes(b"changed during sampling")
        return response

    changed[8].execute = change_image
    with pytest.raises(ValueError):
        call(changed, changed[4])
    assert changed[7]["denoise_calls_completed"] == 35 and changed[7]["responses_sent"] == 0
    failed = setup(failure_dir)

    def fail(_sample, _batch):
        failed[5].calls.extend({"completed": True} for _ in range(3))
        raise RuntimeError("partial denoise")

    failed[8].execute = fail
    with pytest.raises(RuntimeError, match="partial denoise"):
        call(failed, failed[4])
    assert failed[7]["model_requests_attempted"] == 1 and failed[7]["model_requests_completed"] == 0
    assert failed[7]["denoise_calls_completed"] == 3 and failed[7]["responses_sent"] == 0
