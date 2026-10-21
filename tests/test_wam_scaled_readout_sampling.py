"""028 artifact admission and generation delegation without WAM or episode reads."""

import copy
import importlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sampling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("sample_wam_scaled_readout")


def model_evidence(sampling, tmp_path, *, weights_dtype=np.float64, scale=4.0):
    weights = np.zeros((1281, 10), dtype=weights_dtype)
    weights[0, 0], weights[-1, 0] = 8, 2
    center = np.zeros(1280, dtype=np.float64)
    center[0] = 2
    model = tmp_path / "adapt.npz"
    np.savez(model, weights=weights, center=center, scale=np.asarray(scale, dtype=np.float64))
    model_ref = sampling.reference(model)
    fit = tmp_path / "fit.json"
    fit.write_text(
        json.dumps(
            dict(
                eligible_for_runtime_qualification=True,
                eligible_for_sampling=False,
                models=dict(adapt=model_ref),
            )
        )
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(dict(eligible_for_runtime_qualification=True, eligible_for_sampling=False)))
    return dict(
        scaled_fit_ref=sampling.reference(fit),
        scaled_fit_verification_ref=sampling.reference(audit),
        adapted_model_ref=model_ref,
    )


def test_loaded_fp64_center_and_scale_reach_actual_torch_application(sampling, tmp_path, monkeypatch):
    import torch

    from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext

    prepared = model_evidence(sampling, tmp_path)
    calls = []
    monkeypatch.setattr(sampling, "sampling_admission", lambda *args: calls.append(args))
    args = (prepared, tmp_path / "prepared.json", tmp_path / "precision.json", "p", tmp_path / "peer.json", "v")
    readout = sampling.admit_model(*args)
    assert calls == [args]
    assert readout.weights.dtype == readout.center.dtype == np.float64
    assert readout.scale == 4.0 and readout.center[0] == 2.0
    context = DomainContext(LIBERO_DOMAIN, 31, 0, tuple(ACTIVE_MASK))
    hidden = torch.zeros((40, 1280), dtype=torch.bfloat16)
    base = torch.full((40, 128), 0.25, dtype=torch.bfloat16)
    result = readout.apply_torch(hidden, base, context, torch.zeros(40, dtype=torch.long), 0.5)
    expected = base.clone()
    expected[:, 0] = 4.25
    assert torch.equal(result.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(result[:, 10:].view(torch.int16), base[:, 10:].view(torch.int16))
    zero = sampling.ScaledFlowReadout.zero(center=readout.center, scale=readout.scale)
    assert zero.apply_torch(hidden, base, context, torch.zeros(40, dtype=torch.long), 0.5) is base


def test_precision_rejection_precedes_any_model_archive_read(sampling, tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError("independent precision admission rejected")

    def forbidden(*args, **kwargs):
        pytest.fail("model archive opened before precision admission")

    monkeypatch.setattr(sampling, "sampling_admission", reject)
    monkeypatch.setattr(sampling.np, "load", forbidden)
    with pytest.raises(ValueError, match="precision admission rejected"):
        sampling.admit_model({}, tmp_path / "p", tmp_path / "q", "q", tmp_path / "v", "v")


def test_original027_sampling_false_is_preserved_and_required(sampling, tmp_path, monkeypatch):
    prepared = model_evidence(sampling, tmp_path)
    audit_path = Path(prepared["scaled_fit_verification_ref"]["path"])
    audit = json.loads(audit_path.read_text())
    audit["eligible_for_sampling"] = True
    audit_path.write_text(json.dumps(audit))
    prepared["scaled_fit_verification_ref"] = sampling.reference(audit_path)
    monkeypatch.setattr(sampling, "sampling_admission", lambda *args: None)
    with pytest.raises(ValueError, match="original samplingFalse"):
        sampling.admit_model(prepared, tmp_path / "p", tmp_path / "q", "q", tmp_path / "v", "v")


@pytest.mark.parametrize("dtype,scale,message", [(np.float32, 4.0, "FP64"), (np.float64, 0.0, "positive")])
def test_malformed_deployment_model_is_rejected(sampling, tmp_path, monkeypatch, dtype, scale, message):
    prepared = model_evidence(sampling, tmp_path, weights_dtype=dtype, scale=scale)
    monkeypatch.setattr(sampling, "sampling_admission", lambda *args: None)
    with pytest.raises(ValueError, match=message):
        sampling.admit_model(prepared, tmp_path / "p", tmp_path / "q", "q", tmp_path / "v", "v")


def test_generation_request_delegates_full_domain_name_and_actual_seed(sampling):
    calls = []

    class Bridge:
        @contextmanager
        def request(self, context, **kwargs):
            calls.append((context, kwargs))
            yield

    proxy = sampling.RequestBridge(Bridge())
    with pytest.raises(ValueError, match="unarmed"):
        proxy.request("domain")
    proxy.entry = dict(name="actual-name", sampling_seed=43)
    with proxy.request("actual-full-domain"):
        pass
    assert calls == [("actual-full-domain", dict(name="actual-name", seed=43))]


def test_zero_requires_all40x128_raw_and_normalized_values_exact(sampling):
    zeros = np.zeros((40, 128)).tolist()
    response = dict(outputs=[dict(content=dict(action=zeros, action128=dict(normalized_action=zeros)))])
    sampling.check_zero(response, copy.deepcopy(response))
    changed = copy.deepcopy(response)
    changed["outputs"][0]["content"]["action128"]["normalized_action"][-1][-1] = 1e-30
    with pytest.raises(ValueError, match="complete raw/normalized"):
        sampling.check_zero(changed, response)
    short = copy.deepcopy(response)
    short["outputs"][0]["content"]["action"] = zeros[:8]
    with pytest.raises(ValueError, match="complete raw/normalized"):
        sampling.check_zero(short, response)


def test_cpu_environment_is_rejected_before_preparation_or_model_load(sampling, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(
        "sys.argv",
        [
            "sample_wam_scaled_readout.py",
            "--prepared",
            "p",
            "--prepared-sha256",
            "p",
            "--precision",
            "q",
            "--precision-sha256",
            "q",
            "--precision-verification",
            "v",
            "--precision-verification-sha256",
            "v",
            "--seed",
            "42",
        ],
    )
    monkeypatch.setattr(sampling, "read_prepared", lambda *args: pytest.fail("preparation read under invalid CUDA"))
    with pytest.raises(ValueError, match="only physicalGPU6"):
        sampling.main()
