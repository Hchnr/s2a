"""The real denoise guard must allow five scopes while enforcing both limits."""

from types import SimpleNamespace

import pytest
import torch

from scripts.wam.tracking.preview.reset_multitask_counter import make_counter
from scripts.wam.tracking.preview.reset_multitask_live_requests import TASKS
from tests.research.wam.test_reset_multitask_live_requests import Service, call, setup


class Denoiser:
    def denoise(self, *, data_batch_packed):
        return {"preds_action": [torch.zeros((40, 128))]}


class ServiceWithRealCounter(Service):
    def __init__(self, task, model, counter, evidence):
        super().__init__(task, counter, evidence)
        self.model = model
        self.packed = SimpleNamespace(action=SimpleNamespace(tokens=[torch.zeros((40, 128))]))

    def execute(self, _sample, _batch):
        for _ in range(35):
            self.model.denoise(data_batch_packed=self.packed)
        self.evidence.append(dict(stage="initial_noise", inactive_rms_before=0.1, inactive_rms_after=0.0))
        values = [[0.0] * 128 for _ in range(40)]
        return SimpleNamespace(
            model_dump=lambda mode: {
                "status": "success",
                "outputs": [{"content": {"action": values, "action128": {"normalized_action": values}}}],
            }
        )


def test_five_service_requests_use_the_real_global_denoise_guard(tmp_path):
    items = list(setup(tmp_path, TASKS[0]))
    model = Denoiser()
    counter = make_counter(model, items[0])
    items[4] = counter
    items[7] = ServiceWithRealCounter(TASKS[0], model, counter, items[5])
    channel = call(items, items[3])
    assert items[6]["model_sequence_complete"] and items[6]["responses_sent"] == len(channel.sent) == 5
    assert len(counter.calls) == 175 and all(row["completed"] for row in counter.calls)
    assert [counter.calls[index]["label"] for index in range(0, 175, 35)] == [
        "synthetic_8",
        "synthetic_16",
        "synthetic_24",
        "synthetic_32",
        "synthetic_40",
    ]
    with pytest.raises(ValueError, match="exhausted budget"), counter.scope("sixth", 35):
        model.denoise(data_batch_packed=items[7].packed)


def test_per_request_scope_still_requires_exactly_35_forwards():
    model = Denoiser()
    counter = make_counter(model, {"max_requests": 5, "denoise_steps": 35})
    packed = SimpleNamespace(action=SimpleNamespace(tokens=[torch.zeros((40, 128))]))
    with pytest.raises(ValueError, match="does not match its registered scope"), counter.scope("short", 35):
        for _ in range(34):
            model.denoise(data_batch_packed=packed)
    with pytest.raises(ValueError, match="five-request"):
        make_counter(Denoiser(), {"max_requests": 6, "denoise_steps": 35})
