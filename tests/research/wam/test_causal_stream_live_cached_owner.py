"""Check the isolated cached-live owner and model bundle without a GPU."""

import copy
import json

import pytest

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import causal_stream_live_cached_admission as admission
from scripts.wam.tracking.preview import run_causal_stream_live_48 as frozen
from scripts.wam.tracking.preview import run_causal_stream_live_cached as owner


def test_owner_context_reuses_frozen_code_without_changing_its_globals(tmp_path):
    original_output, original_simulation = frozen.OUTPUT, frozen.simulation
    context = owner.build_context(output=tmp_path / "new", qa=tmp_path / "qa", dry=tmp_path / "dry")
    assert context.execute.__code__ is frozen.execute.__code__
    assert context.preflight.__code__ is frozen.preflight.__code__
    assert context.execute.__globals__["OUTPUT"] == tmp_path / "new"
    assert context.execute.__globals__["simulation"] is owner.simulation
    assert frozen.OUTPUT == original_output and frozen.simulation is original_simulation


def test_actual_cached_replay_events_bind_every_endpoint():
    branch = prefix.read(
        {
            "path": str(prefix.REPO / "outputs/causal_stream_endpoint_cache_replay_20261111/simulation/result.json"),
            "sha256": "6c144e223ec50031edac3f6a54579142c9c9fb66927bed18a2423ccd21ed9a58",
        }
    )
    branch["runtime"]["endpoint_cache_events"] = branch["endpoint_cache_events"]
    assert owner.check_cache_events(branch) == {"observed": 40, "reused": [88, 96, 104, 112, 120, 120]}
    changed = copy.deepcopy(branch)
    next(event for event in changed["runtime"]["endpoint_cache_events"] if event["kind"] == "reused")[
        "state_sha256"
    ] = "bad"
    with pytest.raises(ValueError, match="latest exact-state"):
        owner.check_cache_events(changed)


def test_cached_model_worker_uses_new_registration_and_original_model_config(tmp_path, monkeypatch):
    inputs = admission.load_inputs()
    root = tmp_path / "run"
    model_dir = root / "producer/model"
    model_dir.mkdir(parents=True)
    qa, dry = tmp_path / "qa.json", tmp_path / "dry.json"
    qa.write_text("{}")
    dry.write_text("{}")
    monkeypatch.setattr(frozen.previous, "file_stats", lambda _paths: {"verified": "stats"})
    context = owner.build_context(output=root, qa=qa, dry=dry, sources=lambda: ["cached-sources"])
    bundle = context.model_bundle(inputs)
    bundle_ref = publish_result(model_dir / "registered_inputs.json", bundle)
    registration = {
        "phase": "producer",
        "inputs": admission.input_receipt(inputs),
        "source_refs": ["cached-sources"],
        "model_bundle_ref": bundle_ref,
    }
    publish_result(root / "producer/registration.json", registration)
    probe_ref = publish_result(root / "probe.json", {"actual_wait": True, "reaped": True, "returncode": 0})
    preflight = {
        "complete": True,
        "passed": True,
        "inputs": admission.input_receipt(inputs),
        "source_refs": ["cached-sources"],
        "gpu_uuid": admission.GPU,
        "source_files_verified": 100,
        "weight_files_verified": 163,
        "qa_ref": prefix.reference(qa),
        "dry_ref": prefix.reference(dry),
        "model_python_sha256": frozen.repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
        "weight_stats": {"verified": "stats"},
        "cv2_probe_ref": probe_ref,
    }
    publish_result(root / "preflight.json", preflight)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", admission.GPU)
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    for key in prefix.THREAD_VARS:
        monkeypatch.setenv(key, "1")
    loaded = context.model_worker_inputs()
    assert loaded["input_receipt"]["cached_config_ref"] == inputs["cached_config_ref"]
    assert loaded["live_config"]["max_model_requests"] == 6


def test_cached_dry_run_launches_no_model_or_physics(tmp_path):
    path = tmp_path / "dry.json"
    context = owner.build_context(output=tmp_path / "run", qa=tmp_path / "qa", dry=path)
    context.dry_run()
    result = json.loads(path.read_text())
    assert result["complete"] and result["passed"]
    assert result["new_model_requests"] == result["new_denoise_calls"] == result["new_physics_steps"] == 0
