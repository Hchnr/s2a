"""026 cached actual-sigma extraction and fail-closed CPU precision admission."""

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
    spec = importlib.util.spec_from_file_location("flow_readout_fit_test", scripts / "fit_wam_flow_readout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sigma_comes_from_actual_float32_cache_not_json_nominal_decimal(stage, tmp_path):
    calls = []
    expected = []
    for sigma in (0.2, 0.5, 0.8):
        array = tmp_path / f"sigma{sigma}.npz"
        values = np.full((40, 1), sigma, dtype=np.float32)
        np.savez(array, sigma_action=values)
        metadata = tmp_path / f"sigma{sigma}.json"
        metadata.write_text(json.dumps(dict(arrays=stage.original.reference(array))))
        calls.append(dict(sigma=sigma, bundle=stage.original.reference(metadata)))
        expected.extend(values[:, 0].astype(np.float64))
    output = stage.actual_sigmas(dict(teacher_calls=calls * 48), stage.original.Evidence())
    np.testing.assert_array_equal(output, np.tile(expected, 48))
    assert output.dtype == np.float64 and output[0] != 0.2
    # A literal JSON decimal expanded to float64 must not replace the cache.
    np.savez(tmp_path / "wrong.npz", sigma_action=np.full((40, 1), 0.2, dtype=np.float64))
    bad = tmp_path / "wrong.json"
    bad.write_text(json.dumps(dict(arrays=stage.original.reference(tmp_path / "wrong.npz"))))
    with pytest.raises(ValueError, match="positive FP32"):
        stage.actual_sigmas(
            dict(teacher_calls=[dict(sigma=0.2, bundle=stage.original.reference(bad))]), stage.original.Evidence()
        )


def precision_fixture(stage, tmp_path):
    paths = {name: tmp_path / f"{name}.json" for name in ("exit", "release", "gpu", "prepared", "admission")}
    paths["prepared"].write_text("original026")
    paths["exit"].write_text(json.dumps(dict(actual_wait=True, automatic_retry=False, returncode=0)))
    paths["release"].write_text(
        json.dumps(dict(reaped=True, returncode=0, original_error=None, only_owned_child_signalled=True))
    )
    paths["gpu"].write_text(json.dumps(dict(physical_gpu=6, memory_mib=0, released=True)))
    record = dict(
        schema_version="s2a.flow-readout-synthetic-cpu-cuda-precision.v1",
        complete=True,
        passed=True,
        prepared_ref=stage.original.reference(paths["prepared"]),
        core_sha256="frozen-core",
        physical_gpu=6,
        actual_gpu_uuid=stage.GPU_UUID,
        cpu_cuda_bitwise_equal=True,
        model_weights_loaded=0,
        wam_denoise_calls=0,
        fit_groups=0,
        episode_reads=0,
        physics_steps=0,
        exit_ref=stage.original.reference(paths["exit"]),
        release_ref=stage.original.reference(paths["release"]),
        gpu_after_ref=stage.original.reference(paths["gpu"]),
    )
    return paths, record


def test_precision_admission_requires_actual_normal_exit_release_and_exact_core(stage, tmp_path):
    paths, record = precision_fixture(stage, tmp_path)
    paths["admission"].write_text(json.dumps(record))
    stage.precision_admission(
        stage.original.reference(paths["admission"]), record["prepared_ref"], "frozen-core", stage.original.Evidence()
    )
    with pytest.raises(ValueError, match="this core"):
        stage.precision_admission(
            stage.original.reference(paths["admission"]),
            record["prepared_ref"],
            "changed-core",
            stage.original.Evidence(),
        )
    record["fit_groups"] = 1
    paths["admission"].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="zero scientific-work budget"):
        stage.precision_admission(
            stage.original.reference(paths["admission"]),
            record["prepared_ref"],
            "frozen-core",
            stage.original.Evidence(),
        )


def test_precision_failure_and_nonreleased_gpu_cannot_enable_any_fit(stage, tmp_path):
    paths, record = precision_fixture(stage, tmp_path)
    record["passed"] = False
    paths["admission"].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="admission is required"):
        stage.precision_admission(
            stage.original.reference(paths["admission"]),
            record["prepared_ref"],
            "frozen-core",
            stage.original.Evidence(),
        )
    record["passed"] = True
    paths["gpu"].write_text(json.dumps(dict(physical_gpu=6, memory_mib=1, released=False)))
    record["gpu_after_ref"] = stage.original.reference(paths["gpu"])
    paths["admission"].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="release is unproven"):
        stage.precision_admission(
            stage.original.reference(paths["admission"]),
            record["prepared_ref"],
            "frozen-core",
            stage.original.Evidence(),
        )


def test_fit_refuses_gpu_environment_before_any_cache_or_target_read(stage, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(ValueError, match="CPU emptyCUDA/LD"):
        stage.execute(SimpleNamespace(seed=42))
