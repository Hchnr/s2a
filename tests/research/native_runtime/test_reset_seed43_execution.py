"""Exercise the reset owner with recorded receipts and simulated worker failure."""

import copy
import json
import os
from pathlib import Path

import pytest

from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_seed43_execution as execution
from tests.research.native_runtime.test_reset_seed43_feedback_audit import BUNDLE, CHECKER, MODEL, PRODUCER, TOOL


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def setup_trial(tmp_path, monkeypatch):
    output, qa, dry = tmp_path / "trial", tmp_path / "qa.json", tmp_path / "dry.json"
    output.mkdir()
    producer, checker = prefix.read(PRODUCER), prefix.read(CHECKER)
    model, bundle = prefix.read(MODEL), prefix.read(BUNDLE)
    model["model_sequence_complete"] = True  # This field was added after the historical recording.
    config = dict(
        output=str(output.relative_to(prefix.REPO)),
        qa=str(qa.relative_to(prefix.REPO)),
        dry=str(dry.relative_to(prefix.REPO)),
        gpu_uuid=model["gpu_uuid"],
        case_id=producer["case"]["case_id"],
        first_window_steps=8,
        live_window_steps=8,
        max_control_steps_per_phase=16,
        max_native_steps_per_phase=400,
        max_models_per_phase=3,
        max_contexts_per_phase=3,
        max_model_initializations=1,
        max_model_requests=1,
        denoise_calls_per_request=35,
        worker_timeout_seconds=40,
        fresh_cases=0,
        fits=0,
        automatic_retry=False,
    )
    inputs = dict(
        config=config,
        case=dict(tool=TOOL),
        prepared={"model_config": {"model_python": "/synthetic/model-python"}},
        source_sha256={},
    )
    refs = [{"source": "fixed"}]
    trial = registered.RegisteredAdaptiveTrial(
        output=output,
        qa=qa,
        dry=dry,
        module="scripts.simulation.runtime.synthetic",
        gpu_uuid=config["gpu_uuid"],
        required_qa_targets=("tests/research/native_runtime/test_reset_seed43_execution.py",),
        load_inputs=lambda: inputs,
        input_receipt=lambda _inputs: {"fixed": "input"},
        source_refs=lambda: refs,
        model_registration=lambda _inputs, _capture: {},
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


def lifecycle(specs, *, failed=False):
    workers = []
    for name, spec in specs.items():
        (prefix.REPO / spec["log"]).write_text("synthetic lifecycle", encoding="utf-8")
        workers.append(
            dict(
                name=name,
                log=spec["log"],
                actual_wait=True,
                reaped=True,
                returncode=-15 if failed and name == "simulation" else 0,
                signal_sent=failed and name == "simulation",
            )
        )
    return dict(workers=workers, passed=not failed, launch_error=None, timed_out=False, interrupted=failed)


def test_actual_receipts_flow_through_closed_producer_and_independent_checker(tmp_path, monkeypatch):
    trial, inputs, producer, checker, model = setup_trial(tmp_path, monkeypatch)

    def producer_workers(specs, *, timeout_seconds):
        assert timeout_seconds == inputs["config"]["worker_timeout_seconds"]
        assert specs["simulation"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert specs["model"]["env"]["CUDA_VISIBLE_DEVICES"] == trial.gpu_uuid
        write(trial.output / "producer/simulation/result.json", producer)
        write(trial.output / "producer/model/result.json", model)
        return lifecycle(specs)

    def checker_worker(specs, *, cwd, timeout_seconds):
        assert cwd == prefix.REPO and timeout_seconds == inputs["config"]["worker_timeout_seconds"]
        assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["LD_LIBRARY_PATH"] == ""
        write(trial.output / "checker/simulation/result.json", checker)
        Path(specs[0]["log"]).write_text("synthetic replay", encoding="utf-8")
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
    assert saved["complete"] and saved["model_exchange"] == {"requests": 1, "denoise_calls": 35}
    changed = copy.deepcopy(saved)
    changed["gpu_released"] = False
    write(path, changed)
    with pytest.raises(ValueError, match="fully closed"):
        execution.execute(trial, "checker")
    assert not (trial.output / "checker").exists()
    write(path, saved)
    execution.execute(trial, "checker")
    replay = json.loads((trial.output / "checker/result.json").read_text())
    assert replay["complete"] and replay["passed"]
    assert replay["cumulative_physical_bounds"]["control_steps"] == {"lower": 32, "upper": 32}
    assert replay["cumulative_model_bounds"]["requests_completed"] == {"lower": 1, "upper": 1}
    assert all(value == 0 for value in replay["counts"]["independent_max_absolute_difference"].values())


@pytest.mark.parametrize("partial", [False, True])
def test_interrupted_producer_publishes_bounded_partial_costs_and_never_retries(tmp_path, monkeypatch, partial):
    trial, inputs, _producer, _checker, _model = setup_trial(tmp_path, monkeypatch)

    def interrupted(specs, *, timeout_seconds):
        assert timeout_seconds == inputs["config"]["worker_timeout_seconds"]
        if partial:
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
        raise execution.OwnedSocketPairInterrupted(lifecycle(specs, failed=True))

    monkeypatch.setattr(execution, "run_owned_socket_pair", interrupted)
    with pytest.raises(execution.OwnedSocketPairInterrupted):
        execution.execute(trial, "producer")
    result = json.loads((trial.output / "producer/result.json").read_text())
    assert not result["complete"] and result["interrupted"] and result["gpu_released"]
    assert result["automatic_retry"] is False
    assert result["cumulative_physical_bounds"]["control_steps"] == {
        "lower": 7 if partial else 0,
        "upper": 8 if partial else 16,
    }
    assert result["cumulative_model_bounds"]["requests_completed"] == {"lower": 0, "upper": 1}


@pytest.mark.parametrize("key", ["live_window_steps", "max_model_requests", "automatic_retry", "fits"])
def test_changed_budget_is_rejected_before_starting_workers(tmp_path, monkeypatch, key):
    trial, inputs, _producer, _checker, _model = setup_trial(tmp_path, monkeypatch)
    inputs["config"][key] = {"live_window_steps": 1, "max_model_requests": 2, "automatic_retry": True, "fits": 1}[key]
    with pytest.raises(ValueError, match="bounded two-window"):
        execution.execute(trial, "producer")
    assert not (trial.output / "producer").exists()
