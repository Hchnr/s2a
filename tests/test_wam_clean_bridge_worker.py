"""CPU-only035 math/entry guards and actual-hook protocol fixtures.

The tiny fixture scheduler below exists solely to exercise hook object/storage
checks. It is not a stand-in for Cosmos acceptance or the real UniPC algorithm.
"""

import importlib
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from s2a.sampler_trace import file_sha
from s2a.wam_clean_capture import CleanCaptureBridge, NoiseRoster
from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
worker = importlib.import_module("qualify_wam_clean_bridge")
torch = pytest.importorskip("torch")
protocol = importlib.import_module("test_wam_clean_capture")


class WitnessScheduler:
    clone_velocity = clone_sample = duplicate_conversion = False

    def __init__(self):
        self.sigmas = torch.cat((torch.linspace(0.999, 0.125, 35), torch.zeros(1)))
        self.timesteps = torch.arange(35, 0, -1, dtype=torch.long) * 23
        self.step_index = None
        self.predict_x0 = True
        self.config = SimpleNamespace(prediction_type="flow_prediction", thresholding=False)
        self.conversions = 0

    def convert_model_output(self, model_output, *, sample):
        self.conversions += 1
        self.last_result = sample - self.sigmas[self.step_index] * model_output
        return self.last_result

    def step(self, model_output, *, sample):
        if self.step_index is None:
            self.step_index = 0
        velocity = model_output.clone() if self.clone_velocity else model_output
        state = sample.clone() if self.clone_sample else sample
        result = self.convert_model_output(velocity, sample=state)
        assert result is self.last_result
        if self.duplicate_conversion:
            self.convert_model_output(velocity, sample=state)
        self.step_index += 1
        return result


class WitnessSampler(torch.nn.Module):
    @torch.no_grad()
    def forward(self, velocity_fn, noise, num_steps=35, shift=5.0, seed=None):
        sample_scheduler = [WitnessScheduler()]
        self.scheduler = sample_scheduler[0]
        latent = noise
        for timestep in self.scheduler.timesteps:
            returned = velocity_fn(latent, timestep.reshape(1, 1))
            latent = [self.scheduler.step(returned[0], sample=latent[0].unsqueeze(0)).squeeze(0)]
        return latent


def witness_bridge(*, bias=0.05, sink=None):
    model = protocol.ProtocolModel()
    model.sampler = WitnessSampler()
    path, protocol_path = str(Path(__file__).resolve()), str(Path(protocol.__file__).resolve())
    sources = {name: file_sha(name) for name in (path, protocol_path)}
    bridge = worker.ConversionWitness(
        model,
        noise_roster=NoiseRoster((101,)),
        readout=protocol.readout(bias=bias),
        sink=sink,
        forward_file=path,
        forward_sha256=sources[path],
        sources=sources,
        conversion_source_ref=dict(path=path, sha256=sources[path]),
    )
    return model, bridge


def test_actual_conversion_witness_delegates35_times_and_independent_numpy_audits_returned_velocity():
    audits = []

    def sink(point):
        audits.append(worker.audit_online_point(point, np.full(10, 0.05)))

    model, bridge = witness_bridge(sink=sink)
    try:
        with bridge.request(
            DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK), name="protocol", seed=101, purpose="training"
        ):
            model.generate_samples_from_batch()
        assert bridge.last["complete"] and len(audits) == 35 and any(audits)
        assert model.sampler.scheduler.conversions == 35
        assert all(
            step["conversion_calls"] == 1
            and step["conversion_velocity_is_returned_object"]
            and step["conversion_velocity_is_deployed_storage_view"]
            and step["conversion_returned_original_object"]
            and step["conversion_sample_is_original_storage_view"]
            for step in bridge.last["steps"]
        )
        assert "convert_model_output" not in model.sampler.scheduler.__dict__
    finally:
        bridge.restore()


@pytest.mark.parametrize("failure", ["clone_velocity", "clone_sample", "duplicate_conversion"])
def test_conversion_rejects_unrelated_but_equal_objects_and_second_delegation(failure, monkeypatch):
    monkeypatch.setattr(WitnessScheduler, failure, True)
    model, bridge = witness_bridge()
    try:
        with pytest.raises(ValueError):
            with bridge.request(
                DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK), name="reject", seed=101, purpose="training"
            ):
                model.generate_samples_from_batch()
        assert not bridge.last["complete"]
        assert "convert_model_output" not in model.sampler.scheduler.__dict__
        assert model.sampler.scheduler.conversions == (1 if failure == "duplicate_conversion" else 0)
    finally:
        bridge.restore()


def test_explicit_frozen_mask_delegate_is_admitted_and_restored_by_exact_object():
    model = protocol.ProtocolModel()
    original = model.denoise

    def mask_wrapper(instance, *args, **kwargs):
        return original(*args, **kwargs)

    delegate = MethodType(mask_wrapper, model)
    model.denoise = delegate
    path, protocol_path = str(Path(__file__).resolve()), str(Path(protocol.__file__).resolve())
    sources = {name: file_sha(name) for name in (path, protocol_path)}
    args = dict(
        noise_roster=NoiseRoster((101,)),
        forward_file=protocol_path,
        forward_sha256=sources[protocol_path],
        sources=sources,
    )
    with pytest.raises(ValueError, match="exact denoise delegate"):
        CleanCaptureBridge(model, **args)
    bridge = CleanCaptureBridge(model, denoise_delegate=delegate, **args)
    try:
        with bridge.request(
            DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK), name="mask", seed=101, purpose="training"
        ):
            model.generate_samples_from_batch()
        assert bridge.last["complete"]
    finally:
        bridge.restore()
    assert model.denoise is delegate


def test_numpy_direct_bfloat16_rounding_resolves_midpoint_without_fp32_double_rounding():
    raw = np.ones((40, 10), dtype=np.float32)
    residual = np.full((40, 10), -(2**-8 + 2**-30), dtype=np.float64)
    actual = worker.deployment_gold(raw, residual, 1, "torch.bfloat16")
    np.testing.assert_array_equal(actual, np.full((40, 10), 1 + 2**-7, dtype=np.float32))
    incorrectly_double_rounded = (raw.astype(np.float64) - residual).astype(np.float32)
    assert np.all(incorrectly_double_rounded == 1 + 2**-8)
    for sigma in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="actual sigma"):
            worker.deployment_gold(raw, residual, sigma, "torch.bfloat16")


def test_online_numpy_audit_rejects_using_bfloat16_packed_sample_for_c0():
    model, bridge = witness_bridge()
    try:
        with bridge.request(DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK), name="c0", seed=101, purpose="training"):
            model.generate_samples_from_batch()
        point = bridge.last_points[0]
        arrays = dict(point.arrays)
        arrays["c0_full128"] = arrays["packed_action"].astype(np.float64) - point.metadata["sigma_actual"] * arrays[
            "raw_velocity_full128"
        ].astype(np.float64)
        wrong = dataclass_point(point.metadata, arrays)
        with pytest.raises(ValueError, match="actual FP32 sampler state"):
            worker.audit_online_point(wrong, np.full(10, 0.05))
    finally:
        bridge.restore()


def dataclass_point(metadata, arrays):
    return SimpleNamespace(metadata=metadata, arrays=arrays)


def test_historical_output_drift_is_reported_separately_from_protocol_qualification():
    values = np.zeros((40, 128)).tolist()

    def response(action):
        return dict(outputs=[dict(content=dict(action=action, action128=dict(normalized_action=values)))])

    original = response(values)
    changed = np.zeros((40, 128))
    changed[0, 0] = 1e-4
    assert worker.outputs_exact(original, original)
    assert not worker.outputs_exact(response(changed.tolist()), original)
    with pytest.raises(ValueError, match="output differs"):
        worker.require_exact_outputs(response(changed.tolist()), original)


def test_registered_runtime_rejects_a_forbidden_gpu_before_importing_cosmos(monkeypatch):
    config = dict(gpu_uuid=worker.ALLOWED_GPU_UUIDS[0])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(ValueError, match="single explicit allowed UUID"):
        worker.runtime_environment(dict(config=config), worker.ALLOWED_GPU_UUIDS[0])
    assert "cosmos_framework" not in sys.modules
