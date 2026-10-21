"""Actual CPU hook observations and fail-closed admission before any WAM load."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def sampling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_wam_readout_sampling")


def admitted_files(sampling, folder):
    prepared = folder / "prepared.json"
    prepared.write_text("{}")
    model = folder / "adapt.npz"
    np.savez_compressed(model, weights=np.ones((1281, 10), np.float64))
    fit = folder / "fit.json"
    payload = dict(
        complete=True,
        passed=True,
        eligible_for_sampling=True,
        teacher_gate=dict(passed=True),
        producer_fit_groups=2,
        fit_attempts=2,
        fitted_rows=2880,
        validation_rows=2880,
        prepared_ref=sampling.ref(prepared),
        models=dict(adapt=sampling.ref(model)),
    )
    fit.write_text(json.dumps(payload))
    verified = folder / "verified.json"
    admission = dict(
        phase="fit",
        passed=True,
        eligible_for_sampling=True,
        prepared=sampling.ref(prepared),
        input_sha256={str(fit.resolve()): sampling.sha256(fit), str(model.resolve()): sampling.sha256(model)},
    )
    verified.write_text(json.dumps(admission))
    return prepared, fit, verified, payload, admission


def admit(sampling, files):
    prepared, fit, verified = files[:3]
    return sampling.sampling_admission(prepared, fit, sampling.sha256(fit), verified, sampling.sha256(verified))


def test_sampling_requires_both_teacher_gates_before_loading_any_model(sampling, tmp_path):
    files = admitted_files(sampling, tmp_path)
    _, readout = admit(sampling, files)
    assert readout.weights.shape == (1281, 10)
    producer = copy.deepcopy(files[3])
    producer["eligible_for_sampling"] = False
    files[1].write_text(json.dumps(producer))
    with pytest.raises(ValueError, match="producer teacher gate"):
        admit(sampling, files)
    files[1].write_text(json.dumps(files[3]))
    independent = copy.deepcopy(files[4])
    independent["eligible_for_sampling"] = False
    files[2].write_text(json.dumps(independent))
    with pytest.raises(ValueError, match="independent teacher gate"):
        admit(sampling, files)


def test_sampling_rejects_changed_coefficients_and_unrelated_fit(sampling, tmp_path):
    files = admitted_files(sampling, tmp_path)
    independent = copy.deepcopy(files[4])
    independent["input_sha256"][str(files[1].resolve())] = "0" * 64
    files[2].write_text(json.dumps(independent))
    with pytest.raises(ValueError, match="independent teacher gate"):
        admit(sampling, files)
    files[2].write_text(json.dumps(files[4]))
    Path(files[3]["models"]["adapt"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source SHA changed|changed"):
        admit(sampling, files)


def fake_preparation(torch):
    clean = SimpleNamespace(x0_tokens_vision=[torch.zeros((1, 48, 11, 60, 40), dtype=torch.float32)])
    noise = torch.zeros(48 * 11 * 60 * 40 + 40 * 128)
    mask = noise.clone()
    mask[: 48 * 11 * 60 * 40].reshape(1, 48, 11, 60, 40)[:, :, :1] = 1
    return ([], clean, [[1, 2, 3]], [], [noise], [noise.clone()], [mask], True)


def entry():
    return dict(generation_index=0, name="synthetic", pair_id="pair", variant="raw", sampling_seed=42)


def test_initial_capture_observes_once_without_mutating_actual_sampler_values(sampling, tmp_path):
    torch = pytest.importorskip("torch")
    prepared = fake_preparation(torch)
    seen = []

    def original(batch, seed, negative):
        seen.append((batch, seed, negative))
        return prepared

    model = SimpleNamespace(_prepare_inference_data=original)
    observer = sampling.InitialCapture(model)
    observer.arm(entry(), tmp_path)
    raw = dict(ai_caption=['{"task_title":"synthetic"}'])
    result = model._prepare_inference_data(raw, [42])
    assert result is prepared and seen == [(raw, [42], False)]
    with pytest.raises(ValueError, match="repeated inference"):
        model._prepare_inference_data(raw, [42])
    reference = observer.finish()
    metadata = json.loads(Path(reference["path"]).read_text())
    assert metadata["caption"] == raw["ai_caption"][0] and metadata["text_tokens"] == [[1, 2, 3]]
    with np.load(metadata["arrays"]["path"]) as arrays:
        np.testing.assert_array_equal(arrays["initial_noise"], prepared[4][0].numpy())
        assert arrays["first_condition_latent"].shape == (1, 48, 1, 60, 40)
    observer.remove()
    assert model._prepare_inference_data is original


def test_initial_capture_rejects_unarmed_wrong_seed_or_negative_before_preparation(sampling, tmp_path):
    calls = []
    model = SimpleNamespace(_prepare_inference_data=lambda *args: calls.append(args))
    observer = sampling.InitialCapture(model)
    with pytest.raises(ValueError, match="unexpected"):
        model._prepare_inference_data({}, [42])
    observer.arm(entry(), tmp_path)
    for seed, negative in (([43], False), ([42], True)):
        with pytest.raises(ValueError, match="seed/negative"):
            model._prepare_inference_data({}, seed, negative)
    assert calls == []
    with pytest.raises(ValueError, match="did not observe"):
        observer.finish()


def test_denoise_budget_refuses_unregistered_or_extra_call_before_model(sampling):
    torch = pytest.importorskip("torch")
    calls = []
    prediction = dict(preds_action=[torch.zeros(40, 128)])

    def original(**kwargs):
        calls.append(kwargs)
        return prediction

    model = SimpleNamespace(denoise=original)
    counter = sampling.BoundaryCounter(model, 1)
    packed = SimpleNamespace(action=SimpleNamespace(tokens=[torch.zeros(40, 128)]))
    with pytest.raises(ValueError, match="unregistered forward"):
        model.denoise(data_batch_packed=packed)
    with counter.scope("synthetic:0", 1):
        assert model.denoise(data_batch_packed=packed) is prediction
        with pytest.raises(ValueError, match="unregistered forward"):
            model.denoise(data_batch_packed=packed)
    assert len(calls) == 1 and counter.calls == [dict(index=0, label="synthetic:0", completed=True)]
