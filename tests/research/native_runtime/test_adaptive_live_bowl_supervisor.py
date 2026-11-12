"""Exercise frozen preflight and ownership failures without GPU or physics."""

import json
import sys
from pathlib import Path

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_supervisor as owner
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prefix.reference(path)


def cpu_admission(tmp_path, monkeypatch):
    qa, dry = tmp_path / "qa.json", tmp_path / "dry.json"
    source = {"src/example.py": "a" * 64}
    source_ref = write(tmp_path / "sources.json", {"sources": source})
    inputs, bundle, refs = {"input": 1}, {"model": 1}, [{"path": "source", "sha256": "b" * 64}]
    qa_value = dict(
        complete=True,
        passed=True,
        all_tests_passed=True,
        source_unchanged=True,
        actual_wait=True,
        reaped=True,
        signal_sent=False,
        configured_parallelism=64,
        skipped_count=0,
        xfail_count=0,
        xpass_count=0,
        collection={"collection_errors": []},
        targets=list(owner.REQUIRED_QA_TARGETS),
        source_before_ref=source_ref,
    )
    dry_value = dict(
        complete=True,
        passed=True,
        inputs=inputs,
        source_refs=refs,
        model_bundle_sha256=owner._digest(bundle),
        native_models=0,
        physics_steps=0,
        model_initializations=0,
        model_requests=0,
        denoise_calls=0,
        fresh_cases=0,
    )
    write(qa, qa_value)
    write(dry, dry_value)
    monkeypatch.setattr(owner, "QA", qa)
    monkeypatch.setattr(owner, "DRY", dry)
    monkeypatch.setattr(owner, "snapshot", lambda _root: {"sources": source})
    monkeypatch.setattr(owner, "source_refs", lambda: refs)
    monkeypatch.setattr(owner.admission, "input_receipt", lambda _inputs: inputs)
    monkeypatch.setattr(owner, "model_bundle", lambda _inputs: bundle)
    return qa, dry, qa_value, dry_value, source


def test_matching_cpu_and_dry_admission_preserves_source_identity(tmp_path, monkeypatch):
    _qa, _dry, _qa_value, _dry_value, source = cpu_admission(tmp_path, monkeypatch)
    assert owner.check_cpu_admission({}) == source


@pytest.mark.parametrize("change", ["workers", "skip", "missing_target", "source", "dry_physics", "bundle"])
def test_cpu_admission_rejects_incomplete_or_changed_evidence(tmp_path, monkeypatch, change):
    qa, dry, qa_value, dry_value, _source = cpu_admission(tmp_path, monkeypatch)
    if change == "workers":
        qa_value["configured_parallelism"] = 32
    elif change == "skip":
        qa_value["skipped_count"] = 1
    elif change == "missing_target":
        qa_value["targets"].pop()
    elif change == "source":
        monkeypatch.setattr(owner, "snapshot", lambda _root: {"sources": {"changed": "c" * 64}})
    elif change == "dry_physics":
        dry_value["physics_steps"] = 1
    else:
        dry_value["model_bundle_sha256"] = "c" * 64
    qa.write_text(json.dumps(qa_value))
    dry.write_text(json.dumps(dry_value))
    with pytest.raises(ValueError, match="64-worker CPU regression"):
        owner.check_cpu_admission({})


def model_admission(tmp_path, monkeypatch):
    output = tmp_path / "trial"
    (output / "producer/model").mkdir(parents=True)
    qa, dry = tmp_path / "qa", tmp_path / "dry"
    qa.write_text("qa")
    dry.write_text("dry")
    weights = {}
    for index in range(163):
        path = tmp_path / "weights" / f"{index:03d}"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")
        weights[str(path)] = {"sha256": f"{index:064x}"}
    bundle = dict(
        input_receipt={"case": "synthetic"},
        prepared={"weight_files": weights, "model_config": {"model_python": sys.executable}},
        model_registration={"model_sha256": "b" * 64},
        weight_hashes={path: item["sha256"] for path, item in weights.items()},
    )
    refs = [{"path": "source", "sha256": "c" * 64}]
    bundle_ref = write(output / "producer/model/registered_inputs.json", bundle)
    registration = dict(phase="producer", inputs=bundle["input_receipt"], source_refs=refs, model_bundle_ref=bundle_ref)
    write(output / "producer/registration.json", registration)
    preflight = dict(
        complete=True,
        passed=True,
        inputs=bundle["input_receipt"],
        source_refs=refs,
        qa_ref=prefix.reference(qa),
        dry_ref=prefix.reference(dry),
        model_bundle_sha256=owner._digest(bundle),
        gpu_uuid=owner.engine.GPU,
        weight_files_verified=163,
        weight_stats=owner.model_owner.file_stats(bundle["weight_hashes"]),
        model_python_sha256=owner.repro.sha256(sys.executable),
    )
    write(output / "preflight.json", preflight)
    monkeypatch.setattr(owner, "OUTPUT", output)
    monkeypatch.setattr(owner, "QA", qa)
    monkeypatch.setattr(owner, "DRY", dry)
    monkeypatch.setattr(owner, "source_refs", lambda: refs)
    return output, bundle, registration, preflight


def test_model_worker_loads_only_the_fingerprinted_bundle(tmp_path, monkeypatch):
    _output, bundle, _registration, preflight = model_admission(tmp_path, monkeypatch)
    loaded, actual = owner.model_worker_inputs()
    assert loaded == bundle and actual == preflight


@pytest.mark.parametrize("change", ["bundle", "weight_stat", "gpu", "source"])
def test_model_worker_rejects_changed_bundle_preflight_or_weights(tmp_path, monkeypatch, change):
    output, bundle, registration, preflight = model_admission(tmp_path, monkeypatch)
    if change == "bundle":
        bundle["model_registration"]["model_sha256"] = "d" * 64
        replacement = output / "producer/model/changed_inputs.json"
        registration["model_bundle_ref"] = write(replacement, bundle)
        (output / "producer/registration.json").write_text(json.dumps(registration))
    elif change == "weight_stat":
        Path(next(iter(bundle["weight_hashes"]))).write_bytes(b"longer")
    elif change == "gpu":
        preflight["gpu_uuid"] = "GPU-other"
        (output / "preflight.json").write_text(json.dumps(preflight))
    else:
        monkeypatch.setattr(owner, "source_refs", lambda: [{"path": "other", "sha256": "e" * 64}])
    with pytest.raises(ValueError, match="exact preflight and registered bundle"):
        owner.model_worker_inputs()


def test_worker_failure_publishes_unknown_model_cost_when_receipt_missing(tmp_path, monkeypatch):
    output = tmp_path / "trial"
    output.mkdir()
    inputs = {
        "config": {"gpu_uuid": owner.engine.GPU, "worker_timeout_seconds": 5},
        "prepared": {"model_config": {"model_python": "/synthetic/model-python"}},
    }
    monkeypatch.setattr(owner, "OUTPUT", output)
    monkeypatch.setattr(owner.admission, "load_inputs", lambda: inputs)
    monkeypatch.setattr(owner.admission, "input_receipt", lambda _inputs: {})
    monkeypatch.setattr(owner, "check_cpu_admission", lambda _inputs: {})
    monkeypatch.setattr(owner, "check_preflight", lambda _inputs: {})
    monkeypatch.setattr(owner, "source_refs", lambda: [])
    monkeypatch.setattr(owner, "model_bundle", lambda _inputs: {})
    monkeypatch.setattr(owner.engine, "gpu_observation", lambda: {"memory_mib": 0, "compute_apps": []})
    monkeypatch.setattr(
        owner.physical, "_cpu_environment", lambda _directory: {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    )
    monkeypatch.setattr(
        owner.model_owner, "_model_environment", lambda _directory: {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    )

    def fail_workers(specs, *, timeout_seconds):
        assert timeout_seconds == 5 and specs["simulation"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert specs["model"]["env"]["CUDA_VISIBLE_DEVICES"] == owner.engine.GPU
        records = []
        for name, spec in specs.items():
            (owner.ROOT / spec["log"]).write_text("synthetic failure")
            records.append(
                dict(name=name, log=spec["log"], actual_wait=True, reaped=True, returncode=1, signal_sent=False)
            )
        return dict(workers=records, passed=False)

    monkeypatch.setattr(owner, "run_owned_socket_pair", fail_workers)
    with pytest.raises(ValueError, match="workers must both be reaped"):
        owner.execute("producer")
    result = json.loads((output / "producer/result.json").read_text())
    assert not result["complete"] and result["gpu_released"] is True
    assert (
        result["model_initializations"]
        is result["model_requests_attempted"]
        is result["denoise_calls_attempted"]
        is None
    )
    assert "workers_ref" in result and "available_model_ref" not in result


def test_source_list_includes_the_supervisor_and_required_regression_targets():
    refs = owner.source_refs()
    paths = {Path(row["path"]).relative_to(owner.ROOT).as_posix() for row in refs}
    assert set(owner.REQUIRED_QA_TARGETS) <= paths
    assert "scripts/simulation/runtime/adaptive_live_bowl_supervisor.py" in paths
