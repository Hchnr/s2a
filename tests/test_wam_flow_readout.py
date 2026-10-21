"""026 synthetic FP64 fits and real Torch sigma/request/solver boundaries."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from s2a.sampler_trace import file_sha, rng_identity
from s2a.wam_flow_readout import FlowReadoutHook, FlowResidualReadout, SolverSigmaBridge, design_matrix
from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext

torch = pytest.importorskip("torch")


def domain():
    return DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)


def test_fixed_velocity_objective_matches_augmented_ls_not_sigma_squared_loss():
    rng = np.random.default_rng(42)
    hidden, target = rng.normal(size=(12, 4)), rng.normal(size=(12, 10))
    sigma = np.tile(np.asarray([0.2, 0.5, 0.8], np.float32), 4)
    weights = np.full(12, 1 / 12)
    fitted = FlowResidualReadout.fit(hidden, sigma, target, weights)
    phi = np.column_stack((hidden / 2, np.ones(12)))
    design = -phi / sigma.astype(np.float64)[:, None]
    expected = np.linalg.lstsq(
        np.vstack((np.sqrt(weights)[:, None] * design, np.sqrt(0.1) * np.eye(5))),
        np.vstack((np.sqrt(weights)[:, None] * target, np.zeros((5, 10)))),
        rcond=None,
    )[0]
    np.testing.assert_allclose(fitted.weights, expected, atol=1e-14, rtol=1e-13)
    np.testing.assert_array_equal(design_matrix(hidden, sigma), design)
    # JSON .2 is a different value from the cached FP32 sigma expanded exactly.
    assert design[0, -1] != -1 / 0.2
    bias = FlowResidualReadout.fit_bias(hidden, sigma, target, weights)
    single = design[:, -1:]
    expected_bias = np.linalg.lstsq(
        np.vstack((np.sqrt(weights)[:, None] * single, [[np.sqrt(0.1)]])),
        np.vstack((np.sqrt(weights)[:, None] * target, np.zeros((1, 10)))),
        rcond=None,
    )[0]
    np.testing.assert_allclose(bias.weights[-1:], expected_bias, atol=1e-14, rtol=1e-13)
    np.testing.assert_array_equal(bias.weights[:-1], np.zeros((4, 10)))


def test_actual_torch_sigma_sign_precision_and_identity_inactive_isolation():
    sigma = np.asarray([0.2, 0.8], np.float32)
    hidden = torch.zeros(2, 2, dtype=torch.bfloat16)
    base = torch.ones(2, 128, dtype=torch.bfloat16)
    base[:, 10] = -0.0
    before = base.view(torch.int16).clone()
    weights = np.zeros((3, 10))
    weights[-1] = -0.01
    model = FlowResidualReadout(weights)
    output = model.apply_torch(hidden, base, domain(), torch.zeros(2, dtype=torch.long), sigma)
    assert output[0, 0] > output[1, 0] > base[0, 0]
    assert torch.equal(output[:, 10:].view(torch.int16), before[:, 10:])
    assert torch.equal(base.view(torch.int16), before)
    assert (
        FlowResidualReadout.zero(2).apply_torch(hidden, base, domain(), torch.zeros(2, dtype=torch.long), sigma) is base
    )
    other = DomainContext("other/full/domain", 2, 0, ACTIVE_MASK)
    assert model.apply_torch(hidden, base, other, torch.zeros(2, dtype=torch.long), sigma) is base
    for invalid in (0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="sigma must be finite and positive"):
            FlowResidualReadout.zero(2).apply_torch(hidden, base, domain(), torch.zeros(2, dtype=torch.long), invalid)


def test_flow_sum_has_mathematical_single_bfloat16_rounding():
    weights = np.zeros((2, 10), dtype=np.float64)
    weights[-1] = -0.5 * (2**-8 + 2**-30)
    hidden, base = torch.zeros(1, 1, dtype=torch.bfloat16), torch.ones(1, 128, dtype=torch.bfloat16)
    actual = FlowResidualReadout(weights).apply_torch(
        hidden, base, domain(), torch.zeros(1, dtype=torch.long), np.float32(0.5)
    )
    assert torch.equal(actual[:, :10], torch.full((1, 10), 1 + 2**-7, dtype=torch.bfloat16))
    assert torch.equal(actual[:, 10:], base[:, 10:])


class MiniScheduler:
    invalid_sigma = False
    wrong_step = False

    def __init__(self):
        self.sigmas = torch.linspace(1, 0.1, 35, dtype=torch.float32)
        if self.invalid_sigma:
            self.sigmas[0] = 0
        self.sigmas = torch.cat((self.sigmas, torch.zeros(1)))
        self.timesteps = torch.arange(35, 0, -1, dtype=torch.int64) * 7
        self.step_index = 1 if self.wrong_step else None


class MiniSampler(torch.nn.Module):
    @torch.no_grad()
    def forward(self, velocity_fn, noise, num_steps=35, shift=5.0, seed=None):
        sample_scheduler = [MiniScheduler()]
        latent = noise
        for index, timestep in enumerate(sample_scheduler[0].timesteps):
            output = velocity_fn(latent, timestep.reshape(1, 1))
            latent = [latent[0] - output[0] * 0.01]
            sample_scheduler[0].step_index = index + 1
        self.result = latent
        return latent


class Head(torch.nn.Module):
    def forward(self, hidden, projection_ids):
        return hidden[:, :128].clone()


class MiniModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sampler, self.head = MiniSampler(), Head()
        self.velocity_calls = 0

    def generate_samples_from_batch(self, seed=42):
        guidance, velocity_postprocess, velocity_postprocess_builder = 1.0, None, None  # noqa: F841
        _dp_shard_group, _extra_num_steps, scheduler_type = None, 0, "unipc"  # noqa: F841
        initial_noise = [torch.zeros((1, 128), dtype=torch.bfloat16)]

        def velocity_fn(noise, timestep):
            self.velocity_calls += 1
            return [self.head(torch.zeros((1, 1280), dtype=torch.bfloat16), torch.zeros(1, dtype=torch.long))]

        sampler = self.sampler
        return sampler(velocity_fn, initial_noise, num_steps=35, shift=5.0, seed=[seed])


def bridge_for(model, model_readout=None):
    path = str(Path(__file__).resolve())
    hook = FlowReadoutHook(model.head, model_readout or FlowResidualReadout.zero())
    return SolverSigmaBridge(
        model, hook, forward_file=path, forward_sha256=file_sha(path), sources={path: file_sha(path)}
    ), hook


def test_actual_cpu_sampler_wrapper_delegates_once_uses_solver_sigma_not_timestep():
    model = MiniModel()
    body = inspect.unwrap(model.sampler.forward.__func__)
    bridge, hook = bridge_for(model)
    before_rng = rng_identity()
    with bridge.request(domain(), name="zero", seed=42):
        result = model.generate_samples_from_batch()
    assert result is model.sampler.result
    assert model.velocity_calls == 35 and bridge.last["complete"]
    assert len(bridge.last["steps"]) == 35
    first = bridge.last["steps"][0]
    assert first["solver_sigma"] == 1 and first["solver_timestep"] == 245
    assert first["solver_sigma"] != first["solver_timestep"] / 1000
    assert all(row["head_calls"] == 1 for row in bridge.last["steps"])
    assert before_rng == rng_identity()
    bridge.restore()
    hook.remove()
    assert inspect.unwrap(model.sampler.forward.__func__) is body


@pytest.mark.parametrize("wrong", ["invalid_sigma", "wrong_step"])
def test_bad_actual_solver_step_rejected_before_any_velocity_or_head(wrong, monkeypatch):
    monkeypatch.setattr(MiniScheduler, wrong, True)
    model = MiniModel()
    bridge, hook = bridge_for(model)
    with pytest.raises(ValueError):
        with bridge.request(domain(), name="reject", seed=42):
            model.generate_samples_from_batch()
    assert model.velocity_calls == 0
    assert not bridge.last["complete"]
    bridge.restore()
    hook.remove()


def test_missing_request_step_and_cross_hook_request_are_rejected():
    model = MiniModel()
    bridge, hook = bridge_for(model)
    with pytest.raises(ValueError, match="missing or repeated"):
        model.generate_samples_from_batch()
    with hook.request(domain()):
        with pytest.raises(ValueError, match="actual solver step"):
            model.head(torch.zeros(1, 1280), torch.zeros(1, dtype=torch.long))
        another = FlowReadoutHook(Head(), FlowResidualReadout.zero())
        with pytest.raises(ValueError, match="non-nested"):
            with another.request(domain()):
                pass
        another.remove()
    bridge.restore()
    hook.remove()
