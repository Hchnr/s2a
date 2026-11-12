"""Exercise trial context, CPU admission and GPU preflight without GPU work."""

import json
from pathlib import Path

import pytest

from scripts.simulation.runtime import registered_adaptive_trial as trial_module
from scripts.simulation.runtime import registered_model_bundle as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prefix.reference(path)


def setup_trial(tmp_path, monkeypatch):
    output, qa, dry = tmp_path / "trial", tmp_path / "qa.json", tmp_path / "dry.json"
    source = {"src/example.py": "a" * 64}
    refs = [{"path": "source", "sha256": "b" * 64}]
    weights = {f"weight-{index}": "c" * 64 for index in range(163)}
    config = dict(
        gpu_uuid="GPU-example",
        output=str(output.relative_to(prefix.REPO)),
        qa=str(qa.relative_to(prefix.REPO)),
        dry=str(dry.relative_to(prefix.REPO)),
    )
    inputs = dict(
        config=config,
        refs={"fixture": "not a real preflight"},
        prepared={"model_config": {"model_python": "/fixture/python"}},
        bridge={},
        template={},
        model_source_hashes={f"source-{index}": "d" * 64 for index in range(100)},
        weight_hashes=weights,
    )
    trial = trial_module.RegisteredAdaptiveTrial(
        output=output,
        qa=qa,
        dry=dry,
        module="scripts.simulation.runtime.example",
        gpu_uuid=config["gpu_uuid"],
        required_qa_targets=("tests/research/native_runtime/test_long_adaptive_stream.py",),
        load_inputs=lambda: inputs,
        input_receipt=lambda item: {"config": item["config"]},
        source_refs=lambda: refs,
        model_registration=lambda _item, capture: {"image_root": str(capture)},
    )
    source_ref = write(tmp_path / "sources.json", {"sources": source})
    qa_value = dict(
        complete=True,
        passed=True,
        all_tests_passed=True,
        source_unchanged=True,
        actual_wait=True,
        reaped=True,
        signal_sent=False,
        configured_parallelism=64,
        execution_parallelism=64,
        aggregate_node_partition_exact=True,
        call_count=1,
        skipped_count=0,
        xfail_count=0,
        xpass_count=0,
        collection={"collection_errors": []},
        targets=list(trial.required_qa_targets),
        source_before_ref=source_ref,
    )
    dry_value = dict(
        complete=True,
        passed=True,
        inputs=trial.input_receipt(inputs),
        source_refs=refs,
        model_bundle_sha256=registered._digest(trial.model_bundle(inputs)),
        native_models=0,
        physics_steps=0,
        model_initializations=0,
        model_requests=0,
        denoise_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    write(qa, qa_value)
    write(dry, dry_value)
    monkeypatch.setattr(trial_module, "snapshot", lambda _root: {"sources": source})
    return trial, inputs, source, refs, qa_value, dry_value


def test_gpu_observation_filters_by_registered_uuid(monkeypatch):
    def query(command, **_kwargs):
        if "--query-gpu=uuid,memory.used" in command:
            return "GPU-other, 700\nGPU-selected, 12\n"
        return "GPU-other, 9, 700\nGPU-selected, 17, 12\n"

    monkeypatch.setattr(trial_module.subprocess, "check_output", query)
    assert trial_module.gpu_observation("GPU-selected") == {
        "memory_mib": 12,
        "compute_apps": ["GPU-selected, 17, 12"],
    }
    with pytest.raises(ValueError, match="uniquely visible"):
        trial_module.gpu_observation("GPU-missing")


def test_trial_rejects_paths_or_targets_outside_repo_outputs(tmp_path, monkeypatch):
    trial, _inputs, _source, _refs, _qa, _dry = setup_trial(tmp_path, monkeypatch)
    values = dict(vars(trial))
    values["output"] = Path("/tmp/not-s2a")
    with pytest.raises(ValueError, match="local output"):
        trial_module.RegisteredAdaptiveTrial(**values)
    values = dict(vars(trial))
    values["required_qa_targets"] = ("../not-a-test.py",)
    with pytest.raises(ValueError, match="local output"):
        trial_module.RegisteredAdaptiveTrial(**values)


@pytest.mark.parametrize("change", ["parallelism", "reap", "skip", "partition", "source", "dry", "bundle"])
def test_cpu_gate_rejects_incomplete_or_changed_evidence(tmp_path, monkeypatch, change):
    trial, inputs, source, _refs, qa, dry = setup_trial(tmp_path, monkeypatch)
    assert trial.check_cpu_admission(inputs) == source
    if change == "parallelism":
        qa["execution_parallelism"] = 32
    elif change == "reap":
        qa["reaped"] = False
    elif change == "skip":
        qa["skipped_count"] = 1
    elif change == "partition":
        qa["aggregate_node_partition_exact"] = False
    elif change == "source":
        monkeypatch.setattr(trial_module, "snapshot", lambda _root: {"sources": {"changed": "e" * 64}})
    elif change == "dry":
        dry["physics_steps"] = 1
    else:
        dry["model_bundle_sha256"] = "f" * 64
    trial.qa.write_text(json.dumps(qa))
    trial.dry.write_text(json.dumps(dry))
    with pytest.raises(ValueError, match="64-worker CPU regression"):
        trial.check_cpu_admission(inputs)


def patch_preflight(monkeypatch, inputs, refs):
    stats = {"fixture": {"size": 1}}
    monkeypatch.setattr(trial_module, "gpu_observation", lambda _uuid: {"memory_mib": 0, "compute_apps": []})
    monkeypatch.setattr(
        trial_module.model_owner,
        "verify_model_manifests",
        lambda _inputs: {"source_files_verified": 100, "weight_files_verified": 163, "weight_stats": stats},
    )
    monkeypatch.setattr(trial_module.model_owner, "file_stats", lambda _weights: stats)
    monkeypatch.setattr(trial_module.repro, "sha256", lambda _path: "e" * 64)
    monkeypatch.setattr(
        trial_module.live_cv2_admission,
        "run_cpu_probe",
        lambda _python, _directory: {"actual_wait": True, "reaped": True, "returncode": 0},
    )
    return stats


def test_preflight_binds_sources_weights_probe_and_gpu_without_running_model(tmp_path, monkeypatch):
    trial, inputs, _source, refs, _qa, _dry = setup_trial(tmp_path, monkeypatch)
    patch_preflight(monkeypatch, inputs, refs)
    ref = trial.preflight()
    result = prefix.read(ref)
    assert result["complete"] is result["passed"] is True
    assert result["source_files_verified"] == 100 and result["weight_files_verified"] == 163
    assert result["model_initializations"] == result["model_requests"] == result["physics_steps"] == 0
    assert trial.check_preflight(inputs) == result
    refs.append({"path": "changed", "sha256": "f" * 64})
    with pytest.raises(ValueError, match="source-matched"):
        trial.check_preflight(inputs)


def test_busy_gpu_preserves_failed_preflight_and_does_not_start_model(tmp_path, monkeypatch):
    trial, inputs, _source, refs, _qa, _dry = setup_trial(tmp_path, monkeypatch)
    patch_preflight(monkeypatch, inputs, refs)
    monkeypatch.setattr(trial_module, "gpu_observation", lambda _uuid: {"memory_mib": 1, "compute_apps": []})
    with pytest.raises(ValueError, match="GPU must be idle"):
        trial.preflight()
    failure = prefix.read(prefix.reference(trial.output / "preflight_failed.json"))
    assert failure["complete"] is False and failure["error"]["type"] == "ValueError"
    assert not (trial.output / "preflight.json").exists()
