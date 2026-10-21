"""027 exact control reuse and CPU/source gates with synthetic data only."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def stage(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("scaled_readout_fit_test", scripts / "fit_wam_scaled_readout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def control_sources(stage, tmp_path):
    bias = tmp_path / "old_bias.npz"
    coefficients = np.zeros((1281, 10), dtype=np.float64)
    coefficients[-1] = np.linspace(-0.05, 0.05, 10)
    np.savez(bias, weights=coefficients)
    arrays = dict(base=np.zeros((4, 128), np.float32))
    rows = [dict(endpoint=i) for i in range(4)]
    weights = np.full(4, 0.25)
    sigma = np.full(4, float(np.float32(0.2)))
    predictions = tmp_path / "old_predictions.npz"
    np.savez(
        predictions,
        raw=arrays["base"],
        bias=np.zeros((4, 128), np.float32),
        training_weights=weights,
        sigma_actual=sigma,
    )
    config = dict(train_episodes=list(range(6)), validation_episodes=list(range(6, 12)))
    prepared = dict(
        config=config,
        source_features_ref=dict(path="old144", sha256="old144sha"),
        reused_bias_ref=stage.original.reference(bias),
        reused_bias_predictions_ref=stage.original.reference(predictions),
    )
    old = dict(
        complete=True,
        passed=True,
        eligible_for_sampling=False,
        teacher_gate=dict(passed=False),
        features_ref=prepared["source_features_ref"],
        teacher_rows=rows,
        training_episodes=config["train_episodes"],
        validation_episodes=config["validation_episodes"],
        ridge=0.1,
        deployment_dtype="torch.bfloat16",
        models=dict(bias=prepared["reused_bias_ref"]),
        teacher_predictions=prepared["reused_bias_predictions_ref"],
    )
    return prepared, old, arrays, rows, weights, sigma, coefficients


def test_original_control_coefficients_rows_weights_and_actual_sigmas_are_bound(stage, tmp_path):
    prepared, old, arrays, rows, weights, sigma, coefficients = control_sources(stage, tmp_path)
    actual, predictions = stage.reused_control(prepared, old, arrays, rows, weights, sigma, stage.original.Evidence())
    np.testing.assert_array_equal(actual, coefficients)
    assert predictions.shape == (4, 128)
    for changed_weights, changed_sigma in ((weights * 2, sigma), (weights, np.full(4, 0.2))):
        with pytest.raises(ValueError, match="exact source/prediction differs"):
            stage.reused_control(prepared, old, arrays, rows, changed_weights, changed_sigma, stage.original.Evidence())


def test_reuse_rejects_nonzero_hidden_control_or_changed_whole_episode_source(stage, tmp_path):
    prepared, old, arrays, rows, weights, sigma, coefficients = control_sources(stage, tmp_path)
    changed = dict(old, teacher_rows=list(reversed(rows)))
    with pytest.raises(ValueError, match="source or row order differs"):
        stage.reused_control(prepared, changed, arrays, rows, weights, sigma, stage.original.Evidence())
    coefficients[0, 0] = 1
    path = Path(prepared["reused_bias_ref"]["path"])
    np.savez(path, weights=coefficients)
    prepared["reused_bias_ref"] = stage.original.reference(path)
    old["models"]["bias"] = prepared["reused_bias_ref"]
    with pytest.raises(ValueError, match="zero hidden coefficients"):
        stage.reused_control(prepared, old, arrays, rows, weights, sigma, stage.original.Evidence())


def test_real_torch_reused026_bias_is_bitwise_same_after027_preprocessing():
    torch = pytest.importorskip("torch")
    from s2a.wam_flow_readout import FlowResidualReadout
    from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext
    from s2a.wam_scaled_readout import ScaledFlowReadout

    torch.manual_seed(42)
    hidden = torch.randn(3, 1280).to(torch.bfloat16)
    base = torch.randn(3, 128).to(torch.bfloat16)
    base[:, 10] = -0.0
    coefficients = np.zeros((1281, 10), dtype=np.float64)
    coefficients[-1] = np.linspace(-0.05, 0.05, 10)
    context, ids = DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK), torch.zeros(3, dtype=torch.long)
    sigma = np.asarray([0.2, 0.5, 0.8], np.float32).astype(np.float64)
    old = FlowResidualReadout(coefficients).apply_torch(hidden, base, context, ids, sigma)
    transformed = ScaledFlowReadout(coefficients, np.full(1280, 1000), 0.01).apply_torch(
        hidden, base, context, ids, sigma
    )
    assert torch.equal(old.view(torch.int16), transformed.view(torch.int16))


def test_exact_control_comparison_includes_signed_zero_bits(stage):
    with pytest.raises(ValueError, match="exact source/prediction differs"):
        stage.exact_bits(np.asarray([0.0], np.float32), np.asarray([-0.0], np.float32), "signed zero")


def test_cpu_contract_rejects_before_loading_sources_or_fitting(stage, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(ValueError, match="CPU emptyCUDA/LD"):
        stage.execute(SimpleNamespace(seed=42))


def test_registered_plan_config_pin_and_existing_output_are_immutable(stage, tmp_path):
    assert stage.original.sha256(stage.CONFIG) == stage.CONFIG_SHA
    assert stage.original.sha256(stage.PLAN) == stage.PLAN_SHA
    path = tmp_path / "output.json"
    stage.original.write_new(path, dict(candidate_fits=1))
    with pytest.raises(FileExistsError):
        stage.original.write_new(path, dict(candidate_fits=2))
    assert json.loads(path.read_text()) == dict(candidate_fits=1)
