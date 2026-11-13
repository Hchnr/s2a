"""Check the shared adaptive case owner using recorded physical/model receipts."""

import copy
import json
import os
from pathlib import Path

import pytest

from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_multitask_adaptive_execution as execution
from scripts.simulation.runtime.reset_seed43_costs import zero_bounds
from tests.research.native_runtime.test_reset_adaptive_audit import saved_receipts


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def setup_trial(tmp_path, monkeypatch):
    output, qa, dry = tmp_path / "case", tmp_path / "qa.json", tmp_path / "dry.json"
    output.mkdir()
    producer, checker, model, bundle, config = saved_receipts()
    config.update(
        output=str(output.relative_to(prefix.REPO)),
        qa=str(qa.relative_to(prefix.REPO)),
        dry=str(dry.relative_to(prefix.REPO)),
        case_index=2,
        gpu_uuid=model["gpu_uuid"],
        max_native_steps_per_phase=400,
        max_model_initializations=1,
        denoise_calls_per_request=35,
        max_denoise_calls=70,
        worker_timeout_seconds=40,
    )
    physical, model_costs = zero_bounds()
    inputs = dict(
        config=config,
        prepared={"model_config": {"model_python": "/synthetic/python"}},
        previous_physical_bounds=physical,
        previous_model_bounds=model_costs,
        source_sha256={},
    )
    sources = [{"source": "fixed"}]
    trial = registered.RegisteredAdaptiveTrial(
        output=output,
        qa=qa,
        dry=dry,
        module="scripts.simulation.runtime.synthetic",
        gpu_uuid=config["gpu_uuid"],
        required_qa_targets=("tests/research/native_runtime/test_reset_multitask_adaptive_execution.py",),
        load_inputs=lambda: inputs,
        input_receipt=lambda _inputs: {"case": 2},
        source_refs=lambda: sources,
        model_registration=lambda _inputs, _root: {},
    )
    monkeypatch.setattr(registered.RegisteredAdaptiveTrial, "check_cpu_admission", lambda _self, _inputs: {})
    monkeypatch.setattr(registered.RegisteredAdaptiveTrial, "check_preflight", lambda _self, _inputs: {})
    monkeypatch.setattr(registered.RegisteredAdaptiveTrial, "model_bundle", lambda _self, _inputs: bundle)
    monkeypatch.setattr(execution, "snapshot", lambda _root: {"sources": {}})
    monkeypatch.setattr(registered, "gpu_observation", lambda _uuid: {"memory_mib": 0, "compute_apps": []})
    monkeypatch.setattr(
        execution.physical, "_cpu_environment", lambda _directory: {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    )
    monkeypatch.setattr(
        execution.model_owner,
        "_model_environment",
        lambda _directory: {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
    )
    return trial, inputs, producer, checker, model


def workers(specs, *, failed=False):
    rows = []
    for name, spec in specs.items():
        (prefix.REPO / spec["log"]).write_text("synthetic owner worker", encoding="utf-8")
        rows.append(
            dict(
                name=name,
                log=spec["log"],
                actual_wait=True,
                reaped=True,
                returncode=-15 if failed and name == "simulation" else 0,
                signal_sent=failed and name == "simulation",
            )
        )
    return dict(workers=rows, passed=not failed, launch_error=None, timed_out=False, interrupted=failed)


def test_owner_rechecks_producer_before_replay_and_preserves_real_metrics(tmp_path, monkeypatch):
    trial, inputs, producer, checker, model = setup_trial(tmp_path, monkeypatch)

    def producer_workers(specs, *, timeout_seconds):
        assert timeout_seconds == 40
        assert specs["simulation"]["argv"][-4:] == ["--case-index", "2", "--worker", "producer"]
        assert specs["simulation"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert specs["model"]["env"]["CUDA_VISIBLE_DEVICES"] == trial.gpu_uuid
        write(trial.output / "producer/simulation/result.json", producer)
        write(trial.output / "producer/model/result.json", model)
        return workers(specs)

    def checker_worker(specs, *, cwd, timeout_seconds):
        assert cwd == prefix.REPO and timeout_seconds == 40
        assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["LD_LIBRARY_PATH"] == ""
        assert specs[0]["argv"][-4:] == ["--case-index", "2", "--worker", "checker"]
        write(trial.output / "checker/simulation/result.json", checker)
        Path(specs[0]["log"]).write_text("synthetic checker", encoding="utf-8")
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

    monkeypatch.setattr(execution, "run_owned_socket_pair", producer_workers)
    monkeypatch.setattr(execution, "run_owned_workers", checker_worker)
    execution.execute(trial, "producer")
    path = trial.output / "producer/result.json"
    saved = json.loads(path.read_text())
    assert saved["complete"] and saved["counts"]["control_steps"] == 16
    assert saved["model_exchange"] == {"requests": 1, "denoise_calls": 35}
    changed = copy.deepcopy(saved)
    changed["cumulative_physical_bounds"]["control_steps"]["upper"] += 1
    write(path, changed)
    with pytest.raises(ValueError, match="costs"):
        execution.execute(trial, "checker")
    assert not (trial.output / "checker").exists()
    write(path, saved)
    execution.execute(trial, "checker")
    result = json.loads((trial.output / "checker/result.json").read_text())
    assert result["complete"] and result["passed"] and not result["counts"]["stable_last40"]
    assert result["cumulative_physical_bounds"]["control_steps"] == {"lower": 32, "upper": 32}
    assert result["cumulative_model_bounds"]["requests_completed"] == {"lower": 1, "upper": 1}
    assert all(value == 0 for value in result["counts"]["independent_max_absolute_difference"].values())


def test_interrupted_case_preserves_partial_physical_and_unknown_model_work(tmp_path, monkeypatch):
    trial, _inputs, _producer, _checker, _model = setup_trial(tmp_path, monkeypatch)

    def interrupted(specs, *, timeout_seconds):
        assert timeout_seconds == 40
        write(
            trial.output / "producer/simulation/result.json",
            dict(
                complete=False,
                runtime=dict(
                    control_steps_attempted=8,
                    control_steps_completed=7,
                    counters={"mj_step": dict(attempted=198, completed=197)},
                ),
            ),
        )
        raise execution.OwnedSocketPairInterrupted(workers(specs, failed=True))

    monkeypatch.setattr(execution, "run_owned_socket_pair", interrupted)
    with pytest.raises(execution.OwnedSocketPairInterrupted):
        execution.execute(trial, "producer")
    result = json.loads((trial.output / "producer/result.json").read_text())
    assert not result["complete"] and result["interrupted"] and result["gpu_released"]
    assert result["cumulative_physical_bounds"]["control_steps"] == {"lower": 7, "upper": 8}
    assert result["cumulative_model_bounds"]["requests_completed"] == {"lower": 0, "upper": 2}
