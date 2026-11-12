"""Reject changes to registered model inputs without loading GPU code."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.simulation.runtime import registered_model_bundle as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prefix.reference(path)


def registered(tmp_path):
    output = tmp_path / "trial"
    qa, dry = tmp_path / "qa.json", tmp_path / "dry.json"
    write(qa, {"passed": True})
    write(dry, {"passed": True})
    weights = {}
    for index in range(163):
        path = tmp_path / "weights" / f"{index:03d}"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")
        weights[str(path)] = {"sha256": f"{index:064x}"}
    cosmos = tmp_path / "model_source"
    source_hashes = {str(cosmos / f"source_{index:03d}.py"): f"{index:064x}" for index in range(100)}
    bundle = dict(
        input_receipt={"case": "synthetic"},
        prepared={
            "weight_files": weights,
            "model_config": {"model_python": sys.executable, "cosmos_source_root": str(cosmos)},
        },
        bridge={"source_sha256": source_hashes},
        weight_hashes={path: item["sha256"] for path, item in weights.items()},
    )
    sources = [{"path": "source", "sha256": "a" * 64}]
    bundle_ref = write(output / "producer/model/registered_inputs.json", bundle)
    registration = dict(
        phase="producer", inputs=bundle["input_receipt"], source_refs=sources, model_bundle_ref=bundle_ref
    )
    write(output / "producer/registration.json", registration)
    preflight = dict(
        complete=True,
        passed=True,
        inputs=bundle["input_receipt"],
        source_refs=sources,
        qa_ref=prefix.reference(qa),
        dry_ref=prefix.reference(dry),
        model_bundle_sha256=admission._digest(bundle),
        gpu_uuid="GPU-synthetic",
        source_files_verified=100,
        weight_files_verified=163,
        weight_stats=admission.model_owner.file_stats(bundle["weight_hashes"]),
        model_python_sha256=admission.repro.sha256(sys.executable),
    )
    write(output / "preflight.json", preflight)
    options = dict(output=output, qa_path=qa, dry_path=dry, gpu_uuid="GPU-synthetic", source_refs=lambda: sources)
    return options, bundle, preflight


def test_loads_the_exact_registered_bundle_without_model_imports(tmp_path):
    options, bundle, preflight = registered(tmp_path)
    assert admission.load_registered_model_bundle(**options) == (bundle, preflight)


def test_complete_historical_manifest_is_stable_in_a_fresh_process():
    script = """\
from scripts.simulation.runtime import registered_model_bundle as admission
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as historical

fresh = admission.prior_adaptive_source_refs()
historical.previous_model.bind()
assert fresh == admission.prior_adaptive_source_refs() == historical.source_refs()
"""
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        LP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="src:.",
    )
    subprocess.run([sys.executable, "-c", script], cwd=prefix.REPO, env=env, check=True, timeout=600)


@pytest.mark.parametrize("change", ["source", "gpu", "weight", "model_source_count"])
def test_rejects_changed_sources_device_or_model_metadata(tmp_path, change):
    options, bundle, preflight = registered(tmp_path)
    if change == "source":
        options["source_refs"] = lambda: [{"path": "changed", "sha256": "b" * 64}]
    elif change == "gpu":
        options["gpu_uuid"] = "GPU-other"
    elif change == "weight":
        Path(next(iter(bundle["weight_hashes"]))).write_bytes(b"changed")
    else:
        preflight["source_files_verified"] += 1
        (options["output"] / "preflight.json").write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="exact preflight, registered sources and bundle"):
        admission.load_registered_model_bundle(**options)
