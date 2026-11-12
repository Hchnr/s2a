"""Exercise recovery admission, ownership and conservative cost accounting."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_recovery_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as owner
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from tests.research.native_runtime.test_adaptive_live_bowl_audit import rewrapped


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prefix.reference(path)


def config():
    return prefix.read({"path": str(admission.CONFIG), "sha256": admission.CONFIG_SHA})


def prior_physical():
    return dict(
        control_steps=dict(lower=1531, upper=1646),
        native_steps=dict(lower=38275, upper=41150),
        models=dict(lower=78, upper=81),
        contexts=dict(lower=78, upper=81),
    )


def test_real_model_interpreter_accepts_registered_recovery_bundle_without_gpu(tmp_path):
    inputs = admission.load_inputs()
    refs = owner.source_refs()
    bundle = owner.model_bundle(inputs)
    output = tmp_path / "trial"
    simulation = output / "producer/simulation"
    simulation.mkdir(parents=True)
    bundle["model_registration"]["image_root"] = str(simulation.relative_to(owner.ROOT))

    qa, dry = tmp_path / "qa.json", tmp_path / "dry.json"
    write(qa, {"fixture": "qa"})
    write(dry, {"fixture": "dry"})
    bundle_ref = write(output / "producer/model/registered_inputs.json", bundle)
    write(
        output / "producer/registration.json",
        dict(phase="producer", inputs=bundle["input_receipt"], source_refs=refs, model_bundle_ref=bundle_ref),
    )
    write(
        output / "preflight.json",
        dict(
            complete=True,
            passed=True,
            inputs=bundle["input_receipt"],
            source_refs=refs,
            qa_ref=prefix.reference(qa),
            dry_ref=prefix.reference(dry),
            model_bundle_sha256=owner.registered._digest(bundle),
            gpu_uuid=inputs["config"]["gpu_uuid"],
            source_files_verified=len(inputs["model_source_hashes"]),
            weight_files_verified=len(inputs["weight_hashes"]),
            weight_stats=owner.model_owner.file_stats(inputs["weight_hashes"]),
            model_python_sha256=owner.repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
        ),
    )
    script = """
import sys
from pathlib import Path
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as owner
from scripts.wam.tracking.preview.adaptive_model_worker import validate_registration

output, qa, dry = map(Path, sys.argv[1:])
bundle, _ = owner.registered.load_registered_model_bundle(
    output=output, qa_path=qa, dry_path=dry,
    gpu_uuid=owner.engine.GPU, source_refs=owner.source_refs,
)
validate_registration(bundle['model_registration'], output)
assert Path(sys.executable).resolve() == Path(bundle['prepared']['model_config']['model_python']).resolve()
assert not any(name in sys.modules for name in ('torch', 'mujoco', 'robosuite', 'libero', 'cosmos_framework'))
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = "src:."
    environment.update({key: "1" for key in prefix.THREAD_VARS})
    subprocess.run(
        [inputs["prepared"]["model_config"]["model_python"], "-c", script, str(output), str(qa), str(dry)],
        cwd=owner.ROOT,
        env=environment,
        check=True,
        timeout=900,
    )


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
        model_bundle_sha256=owner.registered._digest(bundle),
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


def test_new_model_cost_bounds_keep_attempted_and_completed_distinct():
    cfg = config()
    missing = owner.model_cost_bounds(None, config=cfg, receipt_final=False)
    assert missing["requests_attempted"] == dict(lower=0, upper=2, evidence="not_recorded")
    partial = dict(
        model_initialization_attempts=1,
        model_initializations=1,
        model_requests_attempted=2,
        model_requests_completed=1,
        denoise_calls_attempted=49,
        denoise_calls_completed=48,
    )
    bounded = owner.model_cost_bounds(partial, config=cfg, receipt_final=True)
    assert bounded["requests_attempted"] == dict(lower=2, upper=2, evidence="final_receipt")
    assert bounded["requests_completed"] == dict(lower=1, upper=2, evidence="final_receipt")
    assert bounded["denoise_calls_completed"] == dict(lower=48, upper=49, evidence="final_receipt")
    interim = owner.model_cost_bounds(partial, config=cfg, receipt_final=False)
    assert interim["denoise_calls_attempted"] == dict(lower=49, upper=70, evidence="partial_snapshot")
    partial["model_requests_attempted"] = 3
    with pytest.raises(ValueError, match="within the registered budget"):
        owner.model_cost_bounds(partial, config=cfg, receipt_final=True)


def test_checker_does_not_drop_producer_model_costs():
    cfg = config()
    producer = dict(
        model_initialization_attempts=1,
        model_initializations=1,
        model_requests_attempted=2,
        model_requests_completed=2,
        denoise_calls_attempted=70,
        denoise_calls_completed=70,
    )
    zero = {key: 0 for key in producer}
    prior = owner.known_model_bounds(cfg)
    after_producer = owner.add_model_cost_bounds(
        prior, owner.model_cost_bounds(producer, config=cfg, receipt_final=True)
    )
    after_checker = owner.add_model_cost_bounds(
        after_producer, owner.model_cost_bounds(zero, config=cfg, receipt_final=True)
    )
    assert after_checker == dict(
        initializations_completed=dict(lower=10, upper=10),
        requests_attempted=dict(lower=23, upper=23),
        requests_completed=dict(lower=20, upper=20),
        denoise_calls_completed=dict(lower=700, upper=700),
    )


@pytest.mark.parametrize(
    "change", ["source", "physical_unknown", "physical_mismatch", "model_inflight", "model_missing", "error"]
)
def test_complete_phase_cannot_hide_unknown_or_inconsistent_costs(change):
    counts = dict(control_steps=113, native_steps=2825, models=3, contexts=3)
    physical = {key: dict(lower=value, upper=value, evidence="final_receipt") for key, value in counts.items()}
    model = {
        key: dict(lower=1, upper=1, evidence="final_receipt")
        for key in (
            "initializations_attempted",
            "initializations_completed",
            "requests_attempted",
            "requests_completed",
            "denoise_calls_attempted",
            "denoise_calls_completed",
        )
    }
    result = dict(
        phase="producer",
        counts=counts,
        phase_physical_bounds=physical,
        phase_model_bounds=model,
        partial_source_match=True,
    )
    owner.require_complete_phase_costs(result)
    if change == "source":
        result["partial_source_match"] = False
    elif change == "physical_unknown":
        physical["control_steps"]["evidence"] = "not_recorded"
    elif change == "physical_mismatch":
        physical["native_steps"]["lower"] -= 1
    elif change == "model_inflight":
        model["requests_completed"]["upper"] += 1
    elif change == "model_missing":
        model.pop("denoise_calls_attempted")
    else:
        result["model_counter_error"] = {"type": "ValueError"}
    with pytest.raises(ValueError, match="exact final physical and model cost receipts"):
        owner.require_complete_phase_costs(result)


def test_missing_or_unreaped_worker_receipts_keep_full_registered_ranges(tmp_path, monkeypatch):
    directory = tmp_path / "producer"
    directory.mkdir()
    refs = [{"path": "source", "sha256": "a" * 64}]
    monkeypatch.setattr(owner, "source_refs", lambda: refs)
    cfg = config()
    result = {}
    owner._phase_costs(
        result,
        phase="producer",
        directory=directory,
        inputs={"config": cfg},
        sources=refs,
        lifecycle=None,
        prior_bounds=prior_physical(),
        prior_model_bounds=owner.known_model_bounds(cfg),
    )
    assert result["phase_physical_bounds"]["control_steps"] == dict(lower=0, upper=115, evidence="not_recorded")
    assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=1531, upper=1761)
    assert result["cumulative_model_bounds"]["requests_attempted"] == dict(lower=21, upper=23)


def test_reaped_partial_receipts_bound_actual_work_without_erasing_prior_uncertainty(tmp_path, monkeypatch):
    directory = tmp_path / "producer"
    refs = [{"path": "source", "sha256": "a" * 64}]
    monkeypatch.setattr(owner, "source_refs", lambda: refs)
    cfg = config()
    runtime = dict(
        control_steps_attempted=8,
        control_steps_completed=7,
        counters=dict(
            mj_step=dict(attempted=198, completed=197),
            model_construct=dict(attempted=3, completed=3),
            context=dict(attempted=3, completed=2),
        ),
    )
    model = dict(
        model_initialization_attempts=1,
        model_initializations=1,
        model_requests_attempted=1,
        model_requests_completed=1,
        denoise_calls_attempted=35,
        denoise_calls_completed=35,
    )
    write(directory / "simulation/result.json", {"complete": False, "runtime": runtime})
    write(directory / "model/result.json", model)
    lifecycle = dict(workers=[dict(name=name, actual_wait=True, reaped=True) for name in ("simulation", "model")])
    result = {}
    owner._phase_costs(
        result,
        phase="producer",
        directory=directory,
        inputs={"config": cfg},
        sources=refs,
        lifecycle=lifecycle,
        prior_bounds=prior_physical(),
        prior_model_bounds=owner.known_model_bounds(cfg),
    )
    assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=1538, upper=1654)
    assert result["cumulative_physical_bounds"]["native_steps"] == dict(lower=38472, upper=41348)
    assert result["cumulative_model_bounds"]["requests_completed"] == dict(lower=19, upper=19)
    lifecycle["workers"][0]["reaped"] = False
    result = {}
    owner._phase_costs(
        result,
        phase="producer",
        directory=directory,
        inputs={"config": cfg},
        sources=refs,
        lifecycle=lifecycle,
        prior_bounds=prior_physical(),
        prior_model_bounds=owner.known_model_bounds(cfg),
    )
    assert result["phase_physical_bounds"]["control_steps"] == dict(lower=0, upper=115, evidence="not_recorded")


def test_source_identity_is_stable_before_and_after_legacy_binding_in_a_fresh_process():
    script = """\
import sys
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as owner
from scripts.simulation.runtime import run_causal_replan_step107_model as historical

fresh = owner.source_refs()
historical.bind()
assert fresh == owner.source_refs()
assert not any(name in sys.modules for name in ('torch', 'mujoco', 'robosuite', 'libero', 'cosmos_framework'))
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
    subprocess.run([sys.executable, "-c", script], cwd=owner.ROOT, env=env, check=True, timeout=900)


@pytest.mark.parametrize("has_partial", [False, True])
def test_failed_producer_keeps_missing_or_partial_physical_costs(tmp_path, monkeypatch, has_partial):
    output = tmp_path / "trial"
    output.mkdir()
    cfg = config()
    inputs = {
        "config": cfg,
        "prepared": {"model_config": {"model_python": "/synthetic/model-python"}},
        "failed_attempt": {"cumulative_physical_bounds": prior_physical()},
        "source_sha256": {},
    }
    refs = [{"path": "source", "sha256": "a" * 64}]
    monkeypatch.setattr(owner, "OUTPUT", output)
    monkeypatch.setattr(owner.admission, "load_inputs", lambda: inputs)
    monkeypatch.setattr(owner.admission, "input_receipt", lambda _inputs: {})
    monkeypatch.setattr(owner, "check_cpu_admission", lambda _inputs: {})
    monkeypatch.setattr(owner, "check_preflight", lambda _inputs: {})
    monkeypatch.setattr(owner, "source_refs", lambda: refs)
    monkeypatch.setattr(owner, "model_bundle", lambda _inputs: {})
    monkeypatch.setattr(owner.engine, "gpu_observation", lambda: {"memory_mib": 0, "compute_apps": []})
    monkeypatch.setattr(
        owner.physical, "_cpu_environment", lambda _directory: {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    )
    monkeypatch.setattr(
        owner.model_owner, "_model_environment", lambda _directory: {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    )

    def failed_workers(specs, *, timeout_seconds):
        assert timeout_seconds == cfg["worker_timeout_seconds"]
        if has_partial:
            write(
                output / "producer/simulation/result.json",
                dict(
                    complete=False,
                    runtime=dict(
                        control_steps_attempted=8,
                        control_steps_completed=7,
                        counters={"mj_step": dict(attempted=198, completed=197)},
                    ),
                ),
            )
        write(
            output / "producer/model/result.json",
            dict(
                model_initialization_attempts=0,
                model_initializations=0,
                model_requests_attempted=0,
                model_requests_completed=0,
                denoise_calls_attempted=0,
                denoise_calls_completed=0,
            ),
        )
        rows = []
        for name, spec in specs.items():
            (owner.ROOT / spec["log"]).write_text("synthetic failure")
            rows.append(
                dict(
                    name=name,
                    log=spec["log"],
                    actual_wait=True,
                    reaped=True,
                    returncode=-15 if name == "simulation" else 1,
                    signal_sent=name == "simulation",
                )
            )
        return dict(workers=rows, passed=False)

    monkeypatch.setattr(owner, "run_owned_socket_pair", failed_workers)
    with pytest.raises(ValueError, match="Both model and simulation workers"):
        owner.execute("producer")
    result = json.loads((output / "producer/result.json").read_text())
    assert not result["complete"] and not result["passed"] and result["gpu_released"]
    assert result["automatic_retry"] is False
    assert result["cumulative_model_bounds"]["requests_attempted"] == dict(lower=21, upper=21)
    expected = dict(lower=1538, upper=1654) if has_partial else dict(lower=1531, upper=1761)
    assert result["cumulative_physical_bounds"]["control_steps"] == expected


def test_synthetic_producer_and_independent_checker_preserve_all_cost_ranges(tmp_path, monkeypatch):
    output = tmp_path / "trial"
    output.mkdir()
    cfg = config()
    inputs = {
        "config": cfg,
        "prepared": {"model_config": {"model_python": "/synthetic/model-python"}},
        "failed_attempt": {"cumulative_physical_bounds": prior_physical()},
        "source_sha256": {},
    }
    refs = [{"path": "source", "sha256": "a" * 64}]
    monkeypatch.setattr(owner, "OUTPUT", output)
    monkeypatch.setattr(owner.admission, "load_inputs", lambda: inputs)
    monkeypatch.setattr(owner.admission, "input_receipt", lambda _inputs: {"case": cfg["case_id"]})
    monkeypatch.setattr(owner, "check_cpu_admission", lambda _inputs: {})
    monkeypatch.setattr(owner, "check_preflight", lambda _inputs: {})
    monkeypatch.setattr(owner, "source_refs", lambda: refs)
    monkeypatch.setattr(owner, "model_bundle", lambda _inputs: {})
    monkeypatch.setattr(owner, "snapshot", lambda _root: {"sources": {}})
    monkeypatch.setattr(owner.engine, "gpu_observation", lambda: {"memory_mib": 0, "compute_apps": []})
    monkeypatch.setattr(
        owner.physical, "_cpu_environment", lambda _directory: {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    )
    monkeypatch.setattr(
        owner.model_owner, "_model_environment", lambda _directory: {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    )
    _old_config, producer, checker = rewrapped()
    producer["runtime"]["adaptive_stop_frame"]["frame"]["session"] = cfg["session"]
    producer["runtime"]["adaptive_model_exchanges"] = [{}]
    model = dict(
        complete=True,
        passed=True,
        gpu_uuid=cfg["gpu_uuid"],
        model_initialization_attempts=1,
        model_initializations=1,
        model_requests_attempted=1,
        model_requests_completed=1,
        responses_sent=1,
        denoise_calls_attempted=35,
        denoise_calls_completed=35,
    )
    monkeypatch.setattr(owner.simulation, "load_recorded_responses", lambda _branch, _model: ["verified"])

    def producer_workers(specs, *, timeout_seconds):
        assert timeout_seconds == cfg["worker_timeout_seconds"]
        assert specs["simulation"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert specs["model"]["env"]["CUDA_VISIBLE_DEVICES"] == cfg["gpu_uuid"]
        write(output / "producer/simulation/result.json", producer)
        write(output / "producer/model/result.json", model)
        rows = []
        for name, spec in specs.items():
            (owner.ROOT / spec["log"]).write_text("synthetic")
            rows.append(
                dict(name=name, log=spec["log"], actual_wait=True, reaped=True, returncode=0, signal_sent=False)
            )
        return dict(workers=rows, passed=True, launch_error=None, timed_out=False, interrupted=False)

    def checker_worker(specs, *, cwd, timeout_seconds):
        assert cwd == owner.ROOT and timeout_seconds == cfg["worker_timeout_seconds"]
        write(output / "checker/simulation/result.json", checker)
        Path(specs[0]["log"]).write_text("synthetic")
        return dict(
            workers=[
                dict(
                    key="checker",
                    log=str(specs[0]["log"]),
                    actual_wait=True,
                    reaped=True,
                    returncode=0,
                    signal_sent=False,
                )
            ],
            launch_error=None,
        )

    monkeypatch.setattr(owner, "run_owned_socket_pair", producer_workers)
    monkeypatch.setattr(owner, "run_owned_workers", checker_worker)
    owner.execute("producer")
    producer_path = output / "producer/result.json"
    saved_producer = json.loads(producer_path.read_text())
    for key in ("cumulative_physical_bounds", "cumulative_model_bounds"):
        changed = json.loads(json.dumps(saved_producer))
        changed[key][next(iter(changed[key]))]["upper"] += 1
        producer_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="cumulative ranges must preserve"):
            owner.execute("checker")
        assert not (output / "checker").exists()
    producer_path.write_text(json.dumps(saved_producer))
    owner.execute("checker")
    result = json.loads((output / "checker/result.json").read_text())
    assert result["complete"] and result["passed"] and result["gpu_released"]
    assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=1757, upper=1872)
    assert result["cumulative_physical_bounds"]["native_steps"] == dict(lower=43925, upper=46800)
    assert result["cumulative_model_bounds"]["requests_attempted"] == dict(lower=22, upper=22)
    assert result["cumulative_model_bounds"]["requests_completed"] == dict(lower=19, upper=19)
    assert result["cumulative_model_bounds"]["denoise_calls_completed"] == dict(lower=665, upper=665)
    assert all(value == 0 for value in result["counts"]["independent_max_absolute_difference"].values())
