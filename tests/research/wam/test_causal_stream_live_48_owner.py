"""Validate long-stream owner accounting without running physics or a GPU."""

import copy
import json

import pytest

from s2a.infrastructure.publication import publish_result
from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import causal_stream_live_48_admission as admission
from scripts.wam.tracking.preview import run_causal_stream_live_48 as owner


def previous_branches():
    inputs = admission.load_inputs()
    producer = prefix.read(prefix.read(inputs["long_refs"]["prior_live_producer"])["branch_ref"])
    checker = prefix.read(prefix.read(inputs["long_refs"]["prior_live_checker"])["branch_ref"])
    return inputs, producer, checker


def early_stop_branch():
    inputs, producer, checker = previous_branches()
    branch = copy.deepcopy(producer)
    stream = branch["runtime"]["continuation"]
    stream["complete"] = False
    stream["stop"] = {"reason": "quality_no_witness", "at_step": 96, "next_action_dispatched": False}
    frame = make_stop(session="recorded_correct08_seed42_20261111", next_step=96)
    branch["runtime"]["stream_stop_frame"] = {"required": True, "send_attempted": True, "sent": True, "frame": frame}
    model = {
        "complete": True,
        "passed": True,
        "model_initializations": 1,
        "model_requests_attempted": 2,
        "model_requests_completed": 2,
        "responses_sent": 2,
        "denoise_calls_attempted": 70,
        "denoise_calls_completed": 70,
        "model_sequence_complete": False,
        "model_sequence_stop": frame,
    }
    return inputs, branch, checker, model


def test_owner_accepts_explicit_short_stream_and_exact_model_work():
    _inputs, branch, _checker, model = early_stop_branch()
    assert owner.summarize_branch(branch, phase="producer") == {
        "control_steps": 96,
        "native_steps": 2400,
        "continuation_steps": 16,
        "prefix_inner_samples": 2000,
        "continuation_inner_samples": 400,
    }
    owner.check_model_branch(model, branch)
    model["denoise_calls_completed"] = 69
    with pytest.raises(ValueError, match="Model work"):
        owner.check_model_branch(model, branch)


def test_owner_rejects_a_later_action_after_buffer_loss():
    _inputs, branch, _checker, _model = early_stop_branch()
    branch["runtime"]["continuation"]["segments"][0]["continuation"]["periods"][0]["inner_below_buffer_samples"] = 1
    with pytest.raises(ValueError, match="No action may follow"):
        owner.summarize_branch(branch, phase="producer")


def test_existing_independent_checker_counts_and_isolation_are_preserved():
    _inputs, producer, checker = previous_branches()
    counts = owner.summarize_branch(checker, phase="checker", producer=producer)
    assert counts["continuation_steps"] == 16 and counts["native_steps"] == 2400
    checker["candidate_modules_unloaded"] = False
    with pytest.raises(ValueError, match="isolated checker"):
        owner.summarize_branch(checker, phase="checker", producer=producer)


def test_model_bundle_worker_checks_the_long_registration(tmp_path, monkeypatch):
    inputs = admission.load_inputs()
    root = tmp_path / "live"
    model_dir = root / "producer/model"
    model_dir.mkdir(parents=True)
    qa, dry = tmp_path / "qa.json", tmp_path / "dry.json"
    qa.write_text("{}")
    dry.write_text("{}")
    monkeypatch.setattr(owner, "OUTPUT", root)
    monkeypatch.setattr(owner, "QA", qa)
    monkeypatch.setattr(owner, "DRY", dry)
    monkeypatch.setattr(owner, "source_refs", lambda: ["long-sources"])
    monkeypatch.setattr(owner.previous, "file_stats", lambda _paths: {"verified": "stats"})
    bundle = owner.model_bundle(inputs)
    bundle_ref = publish_result(model_dir / "registered_inputs.json", bundle)
    registration = {
        "phase": "producer",
        "inputs": admission.input_receipt(inputs),
        "source_refs": ["long-sources"],
        "model_bundle_ref": bundle_ref,
    }
    publish_result(root / "producer/registration.json", registration)
    probe_ref = publish_result(root / "probe.json", {"actual_wait": True, "reaped": True, "returncode": 0})
    preflight = {
        "complete": True,
        "passed": True,
        "inputs": admission.input_receipt(inputs),
        "source_refs": ["long-sources"],
        "gpu_uuid": admission.GPU,
        "source_files_verified": 100,
        "weight_files_verified": 163,
        "qa_ref": prefix.reference(qa),
        "dry_ref": prefix.reference(dry),
        "model_python_sha256": owner.repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
        "weight_stats": {"verified": "stats"},
        "cv2_probe_ref": probe_ref,
    }
    publish_result(root / "preflight.json", preflight)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", admission.GPU)
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    for key in prefix.THREAD_VARS:
        monkeypatch.setenv(key, "1")
    assert owner.model_worker_inputs()["live_config"]["max_model_requests"] == 6
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="exactly its registered GPU"):
        owner.model_worker_inputs()


def test_long_dry_admission_never_runs_model_or_physics(tmp_path, monkeypatch):
    path = tmp_path / "dry.json"
    monkeypatch.setattr(owner, "DRY", path)
    owner.dry_run()
    result = json.loads(path.read_text())
    assert result["complete"] and result["passed"]
    assert result["new_model_initializations"] == result["new_model_requests"] == 0
    assert result["new_denoise_calls"] == result["new_physics_steps"] == 0


def test_long_model_worker_rejects_an_unregistered_limit_before_initialization(tmp_path):
    directory = tmp_path / "producer/model"
    directory.mkdir(parents=True)
    inputs = {"live_config": {"max_model_requests": 7, "max_denoise_calls": 245}}
    with pytest.raises(ValueError, match="six-window"):
        owner.live_stream_model_48.model_worker(output=tmp_path, loader=lambda: inputs)
    result = json.loads((directory / "result.json").read_text())
    assert result["model_initialization_attempts"] == result["model_requests_attempted"] == 0
