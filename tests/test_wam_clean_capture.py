"""CPU protocol fixtures only: these are not model or task acceptance tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.sampler_trace import file_sha, rng_identity
from s2a.wam_clean_capture import CleanCaptureBridge, NoiseRoster, point_features
from s2a.wam_clean_readout import CleanResidualReadout
from s2a.wam_readout import ACTIVE_MASK, HIDDEN_SIZE, LIBERO_DOMAIN, DomainContext

torch = pytest.importorskip("torch")


def domain():
    return DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)


class ProtocolScheduler:
    zero_sigma = False

    def __init__(self):
        self.sigmas = torch.cat((torch.linspace(0.999, 0.125, 35), torch.zeros(1)))
        if self.zero_sigma:
            self.sigmas[0] = 0
        self.timesteps = torch.arange(35, 0, -1, dtype=torch.long) * 23
        self.step_index = None


class ProtocolSampler(torch.nn.Module):
    @torch.no_grad()
    def forward(self, velocity_fn, noise, num_steps=35, shift=5.0, seed=None):
        sample_scheduler = [ProtocolScheduler()]
        self.scheduler = sample_scheduler[0]
        self.received = []
        latent = noise
        for index, timestep in enumerate(self.scheduler.timesteps):
            returned = velocity_fn(latent, timestep.reshape(1, 1))
            self.received.append(returned)
            latent = [latent[0] - returned[0].float() * 0.001]
            self.scheduler.step_index = index + 1
        self.result = latent
        return latent


class ProtocolHead(torch.nn.Module):
    def forward(self, hidden, projection_ids):
        return torch.ones((40, 128), dtype=hidden.dtype)


class ProtocolModel(torch.nn.Module):
    sample_dtype = torch.float32
    duplicate_head = duplicate_denoise = False
    wrong_ids = nonzero_inactive = False
    velocity_dtype = torch.bfloat16

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Module()
        self.net.llm2action = ProtocolHead()
        self.sampler = ProtocolSampler()
        self.velocities = self.denoises = 0
        self.raw_returns = []

    def denoise(self, *, data_batch_packed):
        self.denoises += 1
        hidden = torch.zeros((40, HIDDEN_SIZE), dtype=torch.bfloat16)
        ids = torch.full((40,), int(self.wrong_ids), dtype=torch.long)
        head = self.net.llm2action(hidden, ids)
        if self.duplicate_head:
            self.net.llm2action(hidden, ids)
        action = head.to(self.velocity_dtype).clone()
        action[:, 10:] = 0
        return dict(preds_action=[action], preds_vision=[torch.ones((1, 1, 2, 2, 2), dtype=self.velocity_dtype)])

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
        if self.duplicate_denoise:
            self.denoise(data_batch_packed=packed)
        returned = [torch.cat((pred["preds_vision"][0].reshape(-1), pred["preds_action"][0].reshape(-1)))]
        self.raw_returns.append(returned)
        return returned

    def generate_samples_from_batch(self, seed=101):
        guidance, velocity_postprocess, velocity_postprocess_builder = 1.0, None, None  # noqa: F841
        _dp_shard_group, _extra_num_steps, scheduler_type = None, 0, "unipc"  # noqa: F841
        initial = torch.full((40, 128), 1.0003, dtype=torch.float32)
        initial[:, 10:] = int(self.nonzero_inactive)
        initial_noise = [torch.cat((torch.ones(8), initial.reshape(-1))).to(self.sample_dtype)]

        def velocity_fn(noise, timestep):
            return self._get_velocity(noise, timestep)

        sampler = self.sampler
        return sampler(velocity_fn, initial_noise, num_steps=35, shift=5.0, seed=[seed])


def make_bridge(model=None, **kwargs):
    model = ProtocolModel() if model is None else model
    path = str(Path(__file__).resolve())
    return model, CleanCaptureBridge(
        model,
        noise_roster=NoiseRoster((101, 102)),
        forward_file=path,
        forward_sha256=file_sha(path),
        sources={path: file_sha(path)},
        **kwargs,
    )


def readout(*, bias=0, feature_terms=()):
    width = HIDDEN_SIZE + 10 + len(feature_terms)
    coefficients = np.zeros((width + 1, 10))
    coefficients[-1] = bias
    return CleanResidualReadout(coefficients, np.zeros(width), 1, 0.1)


@pytest.mark.parametrize("clean_head", [None, readout()])
def test_raw_and_zero_head_preserve_actual_return_objects_and_fp32_sample(clean_head):
    points = []
    model = ProtocolModel()
    original_denoise = model.denoise.__func__
    model, bridge = make_bridge(model, readout=clean_head, sink=points.append)
    before = rng_identity()
    try:
        with bridge.request(domain(), name="raw", seed=101, purpose="training"):
            result = model.generate_samples_from_batch()
        assert result is model.sampler.result
        assert bridge.last["complete"] and len(points) == len(bridge.last_points) == 35
        assert model.velocities == model.denoises == bridge.total_head_calls == 35
        assert all(actual is raw for actual, raw in zip(model.sampler.received, model.raw_returns, strict=True))
        assert all(step["returned_original_velocity_object"] for step in bridge.last["steps"])
        assert all(
            step["head_calls"] == step["denoise_calls"] == step["velocity_delegations"] == 1
            for step in bridge.last["steps"]
        )
        assert rng_identity() == before
        first = points[0]
        sigma = first.metadata["sigma_actual"]
        assert first.metadata["network_timestep"] / 1000 != sigma
        assert first.metadata["action_offset"] == 8 and first.metadata["action_dim_mask"] == ACTIVE_MASK
        sample = first.action("sample_full128_fp32", layout="active10")
        velocity = first.action("raw_velocity_full128", layout="active10")
        assert sample.dtype == np.float32
        assert np.any(sample != first.arrays["packed_action"][:, :10])
        expected = sample.astype(np.float64) - sigma * velocity.astype(np.float64)
        np.testing.assert_array_equal(first.action("c0_full128", layout="active10"), expected)
        assert np.all(first.arrays["head_base_full128"][:, 10:] == 1)
        assert not np.any(first.action("raw_velocity_full128", layout="full128")[:, 10:])
        assert not np.any(first.action("c0_full128", layout="full128")[:, 10:])
        with pytest.raises(ValueError, match="read-only"):
            first.arrays["hidden"][0, 0] = 5
        with pytest.raises(TypeError):
            first.metadata["action_offset"] = 0
        with pytest.raises(ValueError, match="layout"):
            first.action("c0_full128", layout="implicit")
    finally:
        bridge.restore()
    assert "denoise" not in model.__dict__ and "forward" not in model.sampler.__dict__
    assert model.denoise.__func__ is original_denoise
    assert not model.net.llm2action._forward_hooks


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_nonzero_clean_residual_changes_the_velocity_consumed_by_the_sampler(dtype, monkeypatch):
    monkeypatch.setattr(ProtocolModel, "velocity_dtype", dtype)
    model, bridge = make_bridge(readout=readout(bias=0.05))
    baseline = ProtocolModel()
    unadapted_result = baseline.generate_samples_from_batch()
    try:
        with bridge.request(domain(), name="adapted", seed=101, purpose="training"):
            adapted_result = model.generate_samples_from_batch()
        assert not torch.equal(unadapted_result[0], adapted_result[0])
        for point, raw, actual in zip(bridge.last_points, model.raw_returns, model.sampler.received, strict=True):
            assert actual is not raw
            sigma = point.metadata["sigma_actual"]
            expected = torch.full((40, 10), 1 - 0.05 / sigma, dtype=dtype)
            torch.testing.assert_close(actual[0][8:].reshape(40, 128)[:, :10], expected, rtol=0, atol=0)
            assert torch.equal(actual[0][:8].view(torch.uint8), raw[0][:8].view(torch.uint8))
            assert torch.equal(
                actual[0][8:].reshape(40, 128)[:, 10:].contiguous().view(torch.uint8),
                raw[0][8:].reshape(40, 128)[:, 10:].contiguous().view(torch.uint8),
            )
            assert not np.any(point.action("raw_velocity_full128", layout="full128")[:, 10:])
        assert all(not step["returned_original_velocity_object"] for step in bridge.last["steps"])
    finally:
        bridge.restore()


def test_explicit_time_features_are_unscaled_actual_values_not_an_inferred_basis():
    terms = ("sigma_actual", "network_timestep")
    model, bridge = make_bridge(readout=readout(feature_terms=terms), feature_terms=terms)
    try:
        with bridge.request(domain(), name="features", seed=101, purpose="training"):
            model.generate_samples_from_batch()
        point = bridge.last_points[0]
        values = point_features(point, feature_terms=terms)
        assert values.shape == (40, HIDDEN_SIZE + 12)
        np.testing.assert_array_equal(values[:, -2], np.full(40, point.metadata["sigma_actual"]))
        np.testing.assert_array_equal(values[:, -1], np.full(40, point.metadata["network_timestep"]))
        with pytest.raises(ValueError, match="unregistered"):
            point_features(point, feature_terms=("sigma_squared",))
        with pytest.raises(ValueError, match="captured request contract"):
            point_features(point, feature_terms=())
    finally:
        bridge.restore()
    with pytest.raises(ValueError, match="feature contract"):
        make_bridge(readout=readout(), feature_terms=terms)


def test_training_roster_excludes_evaluation_and_request_state_is_isolated():
    for invalid in ((42,), (43, 101), (101, 101), (), (True,), (-1,)):
        with pytest.raises(ValueError):
            NoiseRoster(invalid)
    model, bridge = make_bridge()
    try:
        with pytest.raises(ValueError, match="purpose roster"):
            with bridge.request(domain(), name="leak", seed=42, purpose="training"):
                model.generate_samples_from_batch(seed=42)
        assert model.velocities == 0
        with bridge.request(domain(), name="first", seed=101, purpose="training"):
            with pytest.raises(ValueError, match="non-nested"):
                with bridge.request(domain(), name="nested", seed=102, purpose="training"):
                    pass
            model.generate_samples_from_batch(seed=101)
        first = bridge.last_points
        with bridge.request(domain(), name="second", seed=43, purpose="evaluation"):
            model.generate_samples_from_batch(seed=43)
        assert len(bridge.last_points) == 35 and bridge.last["seed"] == 43
        assert bridge.last_points[0] is not first[0]
        assert first[0].metadata["noise_seed"] == 101
        assert bridge.last_points[0].metadata["noise_seed"] == 43
        assert bridge.last_points[0].metadata["step_index"] == first[0].metadata["step_index"] == 0
    finally:
        bridge.restore()


@pytest.mark.parametrize(
    "failure", ["duplicate_head", "duplicate_denoise", "wrong_ids", "nonzero_inactive", "sample_dtype", "zero_sigma"]
)
def test_protocol_failures_preserve_failure_and_clear_request_state(failure, monkeypatch):
    if failure == "zero_sigma":
        monkeypatch.setattr(ProtocolScheduler, failure, True)
    else:
        monkeypatch.setattr(ProtocolModel, failure, torch.bfloat16 if failure == "sample_dtype" else True)
    model, bridge = make_bridge()
    try:
        with pytest.raises(ValueError):
            with bridge.request(domain(), name="failure", seed=101, purpose="training"):
                model.generate_samples_from_batch()
        assert not bridge.last["complete"] and bridge.current is bridge._step is None
        if failure in ("zero_sigma", "sample_dtype"):
            assert model.velocities == model.denoises == bridge.total_head_calls == 0
    finally:
        bridge.restore()


def test_no_future_label_or_missing_actual_request_can_enter_online_protocol():
    model, bridge = make_bridge()
    try:
        with pytest.raises(ValueError, match="missing or repeated"):
            model.generate_samples_from_batch()
        with pytest.raises(TypeError, match="future"):
            with bridge.request(domain(), name="future", seed=101, purpose="training", future=np.zeros((40, 128))):
                pass
        with pytest.raises(ValueError, match="actual sampler"):
            model.denoise(data_batch_packed=None)
        with pytest.raises(ValueError, match="single35-node"):
            with bridge.request(domain(), name="incomplete", seed=101, purpose="training"):
                pass
        assert model.velocities == model.denoises == 0
    finally:
        bridge.restore()


def test_capture_sink_cannot_silently_advance_rng_and_failure_does_not_leak_state():
    def bad_sink(point):
        torch.rand(1)

    model, bridge = make_bridge(sink=bad_sink)
    try:
        with pytest.raises(ValueError, match="observer changed"):
            with bridge.request(domain(), name="bad-observer", seed=101, purpose="training"):
                model.generate_samples_from_batch()
        assert not bridge.last["complete"] and bridge.current is bridge._step is None
        bridge.sink = None
        with bridge.request(domain(), name="recovered", seed=102, purpose="training"):
            model.generate_samples_from_batch(seed=102)
        assert bridge.last["complete"] and len(bridge.last_points) == 35
        assert bridge.last_points[0].metadata["step_index"] == 0
        assert bridge.last_points[0].metadata["noise_seed"] == 102
    finally:
        bridge.restore()


def test_sink_observes_only_the_current_step_and_its_copies_cannot_change_sampler_state():
    observed = []
    model = ProtocolModel()

    def sink(point):
        index = point.metadata["step_index"]
        assert model.velocities == model.denoises == index + 1
        assert len(model.sampler.received) == index  # the current v has not reached the solver yet
        detached = point.action("sample_full128_fp32", layout="full128").copy()
        detached[:] = 123
        observed.append(index)

    model, bridge = make_bridge(model, sink=sink)
    baseline = ProtocolModel().generate_samples_from_batch()
    try:
        with bridge.request(domain(), name="causal", seed=101, purpose="training"):
            result = model.generate_samples_from_batch()
        assert observed == list(range(35))
        torch.testing.assert_close(result[0], baseline[0], rtol=0, atol=0)
    finally:
        bridge.restore()
