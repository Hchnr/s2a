"""Validate live owner boundaries without initializing a model or GPU."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from s2a.infrastructure.publication import publish_result
from scripts.wam.tracking.preview import causal_stream_live_admission as admission
from scripts.wam.tracking.preview import run_causal_stream_live_wam as owner


def test_manifest_preflight_hashes_registered_rosters_and_detects_midcheck_change():
    inputs = {
        "model_source_hashes": {f"source{i}": "a" * 64 for i in range(100)},
        "weight_hashes": {f"weight{i}": "b" * 64 for i in range(163)},
    }
    verified = []
    changed = {"after": False}

    def stat(_path):
        return SimpleNamespace(st_size=10, st_mtime_ns=2 if changed["after"] else 1)

    def verify(values, *, workers):
        verified.append((len(values), workers))

    result = owner.verify_model_manifests(inputs, verify=verify, stat_fn=stat)
    assert verified == [(100, 32), (163, 32)]
    assert result["source_files_verified"] == 100 and result["weight_files_verified"] == 163

    def change_during_hash(values, *, workers):
        assert workers == 32
        if len(values) == 163:
            changed["after"] = True

    with pytest.raises(ValueError, match="changed during SHA"):
        owner.verify_model_manifests(inputs, verify=change_during_hash, stat_fn=stat)


def test_model_environment_uses_explicit_uuid_and_repo_local_caches(tmp_path):
    env = owner._model_environment(tmp_path)
    assert env["CUDA_VISIBLE_DEVICES"] == admission.GPU and env["LD_LIBRARY_PATH"] == ""
    assert env["MASTER_ADDR"] == "127.0.0.1" and 0 < int(env["MASTER_PORT"]) < 65536
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    for key in ("TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR"):
        assert env[key].startswith(str(tmp_path))


def test_model_worker_reads_owner_bundle_without_faking_cpu_cuda_environment(tmp_path, monkeypatch):
    inputs = admission.load_inputs()
    bundle = owner.model_bundle(inputs)
    assert "producer" not in bundle and "checker" not in bundle
    root = tmp_path / "live"
    model = root / "producer/model"
    model.mkdir(parents=True)
    qa, dry = tmp_path / "qa.json", tmp_path / "dry.json"
    qa.write_text("{}")
    dry.write_text("{}")
    monkeypatch.setattr(owner, "OUTPUT", root)
    monkeypatch.setattr(owner, "QA", qa)
    monkeypatch.setattr(owner, "DRY", dry)
    monkeypatch.setattr(owner, "source_refs", lambda: ["frozen-source"])
    monkeypatch.setattr(owner, "file_stats", lambda _paths: {"verified": "stats"})
    bundle_ref = publish_result(model / "registered_inputs.json", bundle)
    registration = {
        "phase": "producer",
        "inputs": admission.input_receipt(inputs),
        "source_refs": ["frozen-source"],
        "model_bundle_ref": bundle_ref,
    }
    publish_result(root / "producer/registration.json", registration)
    preflight = {
        "complete": True,
        "passed": True,
        "inputs": admission.input_receipt(inputs),
        "source_refs": ["frozen-source"],
        "gpu_uuid": admission.GPU,
        "source_files_verified": 100,
        "weight_files_verified": 163,
        "qa_ref": owner.prefix.reference(qa),
        "dry_ref": owner.prefix.reference(dry),
        "model_python_sha256": owner.repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
        "weight_stats": {"verified": "stats"},
    }
    preflight["cv2_probe_ref"] = publish_result(
        root / "cv2-probe.json", {"actual_wait": True, "reaped": True, "returncode": 0}
    )
    publish_result(root / "preflight.json", preflight)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", admission.GPU)
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    for key in owner.prefix.THREAD_VARS:
        monkeypatch.setenv(key, "1")
    loaded = owner.model_worker_inputs()
    assert loaded["model_sha256"] == inputs["model_sha256"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == admission.GPU
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="exactly its registered GPU"):
        owner.model_worker_inputs()


def test_live_checker_entry_imports_no_candidate_policy():
    code = """
from scripts.wam.tracking.preview import run_causal_stream_live_wam as owner
from s2a.research.native_runtime.streaming_replay import replay_streaming_continuation
owner.physical.prior.checker_isolation()
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_model_admission_failure_publishes_zero_initializations_without_importing_torch(tmp_path, monkeypatch):
    directory = tmp_path / "producer/model"
    directory.mkdir(parents=True)
    monkeypatch.setattr(owner, "OUTPUT", tmp_path)

    def fail():
        raise ValueError("synthetic model admission failure")

    monkeypatch.setattr(owner, "model_worker_inputs", fail)
    with pytest.raises(ValueError, match="synthetic model admission"):
        owner.model_worker()
    result = json.loads((directory / "result.json").read_text())
    assert result["complete"] is result["passed"] is False
    assert result["model_initialization_attempts"] == result["model_initializations"] == 0
    assert result["model_requests_attempted"] == result["denoise_calls_attempted"] == 0


def test_live_dry_admission_records_zero_new_work(tmp_path, monkeypatch):
    dry = tmp_path / "dry-result.json"
    monkeypatch.setattr(owner, "DRY", dry)
    owner.dry_run()
    result = json.loads(dry.read_text())
    assert result["complete"] and result["passed"]
    assert result["new_model_initializations"] == result["new_model_requests"] == 0
    assert result["new_denoise_calls"] == result["new_physics_steps"] == 0
