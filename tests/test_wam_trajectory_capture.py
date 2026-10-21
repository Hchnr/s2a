"""Actual Torch views, transparent original delegates and final packed timestamps."""

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.sampler_trace import file_sha, rng_identity
from s2a.wam_flow_readout import FlowReadoutHook, FlowResidualReadout
from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext
from s2a.wam_trajectory_capture import (
    TrajectoryCaptureBridge,
    capture_packed,
    exact_packed_times,
    packing_config,
    same_sample_view,
)

torch = pytest.importorskip("torch")


def domain():
    return DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)


def test_scheduler_unsqueeze_is_a_storage_view_not_the_original_python_object():
    actual = torch.arange(32, dtype=torch.float32)
    view = actual.unsqueeze(0)
    assert view is not actual and same_sample_view(view, actual)
    assert not same_sample_view(view.clone(), actual)
    assert not same_sample_view(view.to(torch.float64), actual)
    assert not same_sample_view(actual[1:], actual[:-1])


def test_final_packed_times_require_values_shapes_and_original_dtype():
    packed = SimpleNamespace(
        vision=SimpleNamespace(
            tokens=[torch.ones(1, 2)], timesteps=torch.full((2,), 807.0), condition_mask=[torch.zeros(2)]
        ),
        action=SimpleNamespace(
            tokens=[torch.ones(40, 128)], timesteps=torch.full((40,), 807.0), condition_mask=[torch.zeros(40, 1)]
        ),
    )
    arrays, dtypes = capture_packed(packed)
    exact_packed_times(packed, arrays, dtypes)
    packed.action.timesteps = packed.action.timesteps.to(torch.float64)
    with pytest.raises(ValueError, match="packed action time"):
        exact_packed_times(packed, arrays, dtypes)
    packed.action.timesteps = torch.full((40, 1), 807.0)
    with pytest.raises(ValueError, match="packed action time"):
        exact_packed_times(packed, arrays, dtypes)
    packed.action.timesteps = torch.full((40,), 807.7949)
    with pytest.raises(ValueError, match="packed action time"):
        exact_packed_times(packed, arrays, dtypes)


def test_training_pack_configuration_preserves_missing_spatial_compression_default16():
    @dataclass
    class Packing:
        spatial_compression_factor: int = 16
        latent_patch_size: int = 2

    configured = packing_config(Packing, dict(latent_patch_size=4, unrelated_inference_option=True))
    assert configured.spatial_compression_factor == 16
    assert configured.latent_patch_size == 4


class MiniScheduler:
    duplicate = False
    clone_sample = False

    def __init__(self):
        self.sigmas = torch.cat((torch.linspace(0.999, 0.128, 35), torch.zeros(1)))
        self.timesteps = torch.arange(35, 0, -1, dtype=torch.int64) * 23
        self.step_index = None
        self.predict_x0 = True
        self.config = SimpleNamespace(prediction_type="flow_prediction", thresholding=False)
        self.conversions = 0

    def convert_model_output(self, model_output, *, sample):
        self.conversions += 1
        self.last_output = sample - self.sigmas[self.step_index] * model_output
        return self.last_output

    def step(self, model_output, *, sample):
        if self.step_index is None:
            self.step_index = 0
        supplied = sample.clone() if self.clone_sample else sample
        result = self.convert_model_output(model_output, sample=supplied)
        assert result is self.last_output
        if self.duplicate:
            self.convert_model_output(model_output, sample=supplied)
        self.step_index += 1
        return sample - model_output.float() * 0.001


class MiniSampler(torch.nn.Module):
    @torch.no_grad()
    def forward(self, velocity_fn, noise, num_steps=35, shift=5.0, seed=None):
        sample_scheduler = [MiniScheduler()]
        self.scheduler = sample_scheduler[0]
        latent = noise
        for timestep in self.scheduler.timesteps:
            velocity_pred = velocity_fn(latent, timestep.reshape(1, 1))
            latent = [self.scheduler.step(velocity_pred[0], sample=latent[0].unsqueeze(0)).squeeze(0)]
        self.result = latent
        return latent


class Head(torch.nn.Module):
    def forward(self, hidden, projection_ids):
        return torch.ones((40, 128), dtype=hidden.dtype)


class MiniModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Module()
        self.net.llm2action = Head()
        self.sampler = MiniSampler()
        self.denoises = self.velocities = 0

    def denoise(self, *, data_batch_packed):
        self.denoises += 1
        head = self.net.llm2action(torch.zeros((40, 1280), dtype=torch.bfloat16), torch.zeros(40, dtype=torch.long))
        action = head.clone()
        action[:, 10:] = 0
        return dict(preds_action=[action], preds_vision=[torch.ones((1, 1, 2, 2, 2), dtype=torch.bfloat16)])

    def _get_velocity(self, noise, timestep):
        self.velocities += 1
        packed = SimpleNamespace(
            vision=SimpleNamespace(
                tokens=[noise[0][:8].reshape(1, 1, 2, 2, 2).to(torch.bfloat16)],
                timesteps=timestep.reshape(-1).float().expand(2).clone(),
                condition_mask=[torch.zeros(2, 1, 1)],
            ),
            action=SimpleNamespace(
                tokens=[noise[0][8:].reshape(40, 128).to(torch.bfloat16)],
                timesteps=timestep.reshape(-1).float().expand(40).clone(),
                condition_mask=[torch.zeros(40, 1)],
            ),
        )
        pred = self.denoise(data_batch_packed=packed)
        return [torch.cat((pred["preds_vision"][0].reshape(-1), pred["preds_action"][0].reshape(-1)))]

    def generate_samples_from_batch(self, seed=42):
        guidance, velocity_postprocess, velocity_postprocess_builder = 1.0, None, None  # noqa: F841
        _dp_shard_group, _extra_num_steps, scheduler_type = None, 0, "unipc"  # noqa: F841
        initial_noise = [torch.zeros(8 + 40 * 128, dtype=torch.float32)]

        def velocity_fn(noise, timestep):
            return self._get_velocity(noise, timestep)

        sampler = self.sampler
        return sampler(velocity_fn, initial_noise, num_steps=35, shift=5.0, seed=[seed])


def capture_for(model):
    path = str(Path(__file__).resolve())
    points = []

    def sink(record, arrays, metadata):
        points.append((record.copy(), arrays, metadata))
        return dict(path=f"point{record['step_index']}", sha256="synthetic")

    hook = FlowReadoutHook(model.net.llm2action, FlowResidualReadout.zero())
    bridge = TrajectoryCaptureBridge(
        model,
        hook,
        sink=sink,
        forward_file=path,
        forward_sha256=file_sha(path),
        convert_file=path,
        convert_sha256=file_sha(path),
        sources={path: file_sha(path)},
    )
    return bridge, hook, points


def test_capture_real_cpu_values_delegates_once_and_preserves_rng_and_returns():
    model = MiniModel()
    denoise_body = inspect.unwrap(model.denoise.__func__)
    bridge, hook, points = capture_for(model)
    before = rng_identity()
    with bridge.request(domain(), name="free0", seed=42):
        result = model.generate_samples_from_batch()
    assert result is model.sampler.result
    assert model.denoises == model.velocities == model.sampler.scheduler.conversions == 35
    assert bridge.total_head_calls == 35 and len(points) == len(bridge.last_points) == 3
    assert [point[0]["step_index"] for point in points] == [19, 29, 33]
    assert all(
        step["conversion_calls"] == step["velocity_delegations"] == step["head_calls"] == 1
        for step in bridge.last["steps"]
    )
    assert before == rng_identity()
    assert "convert_model_output" not in model.sampler.scheduler.__dict__
    for record, arrays, metadata in points:
        assert arrays["sample"].dtype == arrays["sample_action"].dtype == np.float32
        np.testing.assert_array_equal(arrays["sample"][8:].reshape(40, 128), arrays["sample_action"])
        assert metadata["action_offset"] == 8
        assert metadata["original_dtypes"]["sample"] == "torch.float32"
        assert metadata["original_dtypes"]["velocity"] == "torch.bfloat16"
        assert metadata["original_dtypes"]["converted_action"] == "torch.float32"
        assert metadata["conversion_sample_is_original_storage_view"]
        assert np.all(arrays["base"][:, 10:] == 1) and not np.any(arrays["velocity_action"][:, 10:])
        sigma = torch.tensor(record["solver_sigma"], dtype=torch.float32)
        actual = torch.from_numpy(arrays["sample_action"]) - sigma * torch.from_numpy(arrays["velocity_action"]).to(
            torch.bfloat16
        )
        np.testing.assert_array_equal(arrays["converted_action"], actual.numpy())
    with bridge.request(domain(), name="free1", seed=43):
        second = model.generate_samples_from_batch(seed=43)
    assert second is model.sampler.result
    assert model.denoises == model.velocities == bridge.total_head_calls == 70
    assert len(points) == 6 and len(bridge.last_points) == 3
    assert bridge.last["name"] == "free1" and bridge.last["seed"] == 43
    assert model.sampler.scheduler.conversions == 35
    assert "convert_model_output" not in model.sampler.scheduler.__dict__
    assert before == rng_identity()
    bridge.restore()
    hook.remove()
    assert inspect.unwrap(model.denoise.__func__) is denoise_body


@pytest.mark.parametrize("wrong", ["duplicate", "clone_sample"])
def test_conversion_rejects_second_delegate_or_equal_but_unrelated_sample(wrong, monkeypatch):
    monkeypatch.setattr(MiniScheduler, wrong, True)
    model = MiniModel()
    bridge, hook, _ = capture_for(model)
    with pytest.raises(ValueError):
        with bridge.request(domain(), name="reject", seed=42):
            model.generate_samples_from_batch()
    assert not bridge.last["complete"]
    assert model.sampler.scheduler.conversions == (1 if wrong == "duplicate" else 0)
    assert "convert_model_output" not in model.sampler.scheduler.__dict__
    bridge.restore()
    hook.remove()
