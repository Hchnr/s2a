"""Exercise independent replay and uncertain costs under the reusable owner."""

import copy
import json
from pathlib import Path

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_long_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_recovery_audit as audit
from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as accounting
from scripts.simulation.runtime import registered_adaptive_execution as execution
from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from tests.research.native_runtime.test_adaptive_live_bowl_audit import rewrapped


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prefix.reference(path)


def context(tmp_path, monkeypatch):
    output, qa, dry = tmp_path / "trial", tmp_path / "qa.json", tmp_path / "dry.json"
    output.mkdir()
    config = json.loads(admission.CONFIG.read_text())
    config.update(
        output=str(output.relative_to(prefix.REPO)),
        qa=str(qa.relative_to(prefix.REPO)),
        dry=str(dry.relative_to(prefix.REPO)),
    )
    inputs = dict(
        config=config,
        prepared={"model_config": {"model_python": "/synthetic/model-python"}},
        previous_physical_bounds=config["previous_physical_bounds"],
        previous_model_bounds=accounting.known_model_bounds(config),
        source_sha256={},
    )
    refs = [{"path": "source", "sha256": "a" * 64}]

    def compare_previous(_inputs, branch):
        prior, adaptive = audit.historical.periods(branch, "producer")
        return {"compared_periods": len(prior) + len(adaptive)}

    trial = registered.RegisteredAdaptiveTrial(
        output=output,
        qa=qa,
        dry=dry,
        module="scripts.simulation.runtime.synthetic",
        gpu_uuid=config["gpu_uuid"],
        required_qa_targets=("tests/research/native_runtime/test_long_adaptive_stream.py",),
        load_inputs=lambda: inputs,
        input_receipt=lambda _inputs: {"case": config["case_id"]},
        source_refs=lambda: refs,
        model_registration=lambda _inputs, _capture: {},
        compare_previous=compare_previous,
    )
    monkeypatch.setattr(registered.RegisteredAdaptiveTrial, "check_cpu_admission", lambda _self, _inputs: {})
    monkeypatch.setattr(registered.RegisteredAdaptiveTrial, "check_preflight", lambda _self, _inputs: {})
    monkeypatch.setattr(registered.RegisteredAdaptiveTrial, "model_bundle", lambda _self, _inputs: {})
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
    return trial, inputs, refs


@pytest.mark.parametrize("change", ["control", "native", "requests", "denoise", "prior_physical", "prior_model"])
def test_inconsistent_budgets_stop_before_creating_or_launching_a_phase(tmp_path, monkeypatch, change):
    trial, inputs, _refs = context(tmp_path, monkeypatch)
    config = inputs["config"]
    if change == "control":
        config["max_control_steps_per_phase"] += 1
    elif change == "native":
        config["max_native_steps_per_phase"] -= 1
    elif change == "requests":
        config["max_model_requests"] = 76
    elif change == "denoise":
        config["max_denoise_calls"] -= 1
    elif change == "prior_physical":
        inputs["previous_physical_bounds"]["models"]["upper"] = 1
    else:
        inputs["previous_model_bounds"]["requests_completed"]["upper"] = 0
    with pytest.raises(ValueError):
        execution.execute(trial, "producer")
    assert not (trial.output / "producer").exists()


@pytest.mark.parametrize("has_partial", [False, True])
def test_failed_producer_preserves_previous_uncertainty_and_actual_partial_cost(tmp_path, monkeypatch, has_partial):
    trial, inputs, _refs = context(tmp_path, monkeypatch)

    def failed_workers(specs, *, timeout_seconds):
        assert timeout_seconds == inputs["config"]["worker_timeout_seconds"]
        if has_partial:
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
        write(
            trial.output / "producer/model/result.json",
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
            (prefix.REPO / spec["log"]).write_text("synthetic failure")
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

    monkeypatch.setattr(execution, "run_owned_socket_pair", failed_workers)
    with pytest.raises(ValueError, match="Both model and simulation workers"):
        execution.execute(trial, "producer")
    result = json.loads((trial.output / "producer/result.json").read_text())
    assert not result["complete"] and not result["passed"] and result["gpu_released"]
    assert result["automatic_retry"] is False
    assert result["cumulative_model_bounds"]["requests_attempted"] == dict(lower=23, upper=23)
    expected = dict(lower=1768, upper=1884) if has_partial else dict(lower=1761, upper=2023)
    assert result["cumulative_physical_bounds"]["control_steps"] == expected


def test_interrupted_owner_keeps_reaped_partial_physics_and_unknown_model_range(tmp_path, monkeypatch):
    trial, inputs, _refs = context(tmp_path, monkeypatch)

    def interrupted(specs, *, timeout_seconds):
        assert timeout_seconds == inputs["config"]["worker_timeout_seconds"]
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
        rows = []
        for name, spec in specs.items():
            (prefix.REPO / spec["log"]).write_text("synthetic interruption")
            rows.append(
                dict(name=name, log=spec["log"], actual_wait=True, reaped=True, returncode=-15, signal_sent=True)
            )
        raise execution.OwnedSocketPairInterrupted(
            dict(workers=rows, passed=False, launch_error=None, timed_out=False, interrupted=True)
        )

    monkeypatch.setattr(execution, "run_owned_socket_pair", interrupted)
    with pytest.raises(execution.OwnedSocketPairInterrupted):
        execution.execute(trial, "producer")
    result = json.loads((trial.output / "producer/result.json").read_text())
    assert result["interrupted"] is True and result["complete"] is False and result["gpu_released"] is True
    assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=1768, upper=1884)
    assert result["cumulative_model_bounds"]["requests_attempted"] == dict(lower=23, upper=63)
    assert result["cumulative_model_bounds"]["denoise_calls_completed"] == dict(lower=700, upper=2100)


def test_replay_rechecks_producer_receipts_comparison_and_cumulative_costs(tmp_path, monkeypatch):
    trial, inputs, _refs = context(tmp_path, monkeypatch)
    config = inputs["config"]
    _old_config, producer, checker = rewrapped()
    producer["runtime"]["adaptive_stop_frame"]["frame"]["session"] = config["session"]
    producer["runtime"]["adaptive_model_exchanges"] = [{}]
    model = dict(
        complete=True,
        passed=True,
        gpu_uuid=config["gpu_uuid"],
        model_initialization_attempts=1,
        model_initializations=1,
        model_requests_attempted=1,
        model_requests_completed=1,
        responses_sent=1,
        denoise_calls_attempted=35,
        denoise_calls_completed=35,
    )
    monkeypatch.setattr(execution.simulation, "load_recorded_responses", lambda _branch, _model: ["verified"])

    def producer_workers(specs, *, timeout_seconds):
        assert timeout_seconds == config["worker_timeout_seconds"]
        assert specs["simulation"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert specs["model"]["env"]["CUDA_VISIBLE_DEVICES"] == config["gpu_uuid"]
        write(trial.output / "producer/simulation/result.json", producer)
        write(trial.output / "producer/model/result.json", model)
        rows = []
        for name, spec in specs.items():
            (prefix.REPO / spec["log"]).write_text("synthetic")
            rows.append(
                dict(name=name, log=spec["log"], actual_wait=True, reaped=True, returncode=0, signal_sent=False)
            )
        return dict(workers=rows, passed=True, launch_error=None, timed_out=False, interrupted=False)

    def checker_worker(specs, *, cwd, timeout_seconds):
        assert cwd == prefix.REPO and timeout_seconds == config["worker_timeout_seconds"]
        write(trial.output / "checker/simulation/result.json", checker)
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

    monkeypatch.setattr(execution, "run_owned_socket_pair", producer_workers)
    monkeypatch.setattr(execution, "run_owned_workers", checker_worker)
    execution.execute(trial, "producer")
    path = trial.output / "producer/result.json"
    saved = json.loads(path.read_text())
    for key in ("cumulative_physical_bounds", "cumulative_model_bounds", "previous_comparison"):
        changed = copy.deepcopy(saved)
        if key == "previous_comparison":
            changed[key]["compared_periods"] += 1
        else:
            changed[key][next(iter(changed[key]))]["upper"] += 1
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            execution.execute(trial, "checker")
        assert not (trial.output / "checker").exists()
    path.write_text(json.dumps(saved))
    execution.execute(trial, "checker")
    result = json.loads((trial.output / "checker/result.json").read_text())
    assert result["complete"] and result["passed"] and result["gpu_released"]
    assert result["cumulative_physical_bounds"]["control_steps"] == dict(lower=1987, upper=2102)
    assert result["cumulative_physical_bounds"]["native_steps"] == dict(lower=49675, upper=52550)
    assert result["cumulative_model_bounds"]["requests_attempted"] == dict(lower=24, upper=24)
    assert result["cumulative_model_bounds"]["requests_completed"] == dict(lower=21, upper=21)
    assert result["cumulative_model_bounds"]["denoise_calls_completed"] == dict(lower=735, upper=735)
    assert all(value == 0 for value in result["counts"]["independent_max_absolute_difference"].values())
