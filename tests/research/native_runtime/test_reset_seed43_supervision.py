"""Require closed, source-bound producer evidence before checker dispatch."""

import copy
import json
from types import SimpleNamespace

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_seed43_costs as costs
from scripts.simulation.runtime import reset_seed43_feedback_audit as audit
from scripts.simulation.runtime import reset_seed43_supervision as supervision
from tests.research.native_runtime.test_reset_seed43_feedback_audit import BUNDLE, MODEL, PRODUCER, TOOL


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return prefix.reference(path)


def setup(tmp_path):
    output = tmp_path / "trial"
    directory = output / "producer"
    branch = prefix.read(PRODUCER)
    model, bundle = prefix.read(MODEL), prefix.read(BUNDLE)
    model["model_sequence_complete"] = True
    branch_ref = write(directory / "simulation/result.json", branch)
    model_ref = write(directory / "model/result.json", model)
    bundle_ref = write(directory / "model/registered_inputs.json", bundle)
    sources = [{"source": "fixed"}]
    inputs = dict(
        case=dict(tool=TOOL),
        config=dict(
            case_id=branch["case"]["case_id"],
            gpu_uuid=model["gpu_uuid"],
            first_window_steps=8,
            live_window_steps=8,
            max_control_steps_per_phase=16,
            max_native_steps_per_phase=400,
            max_models_per_phase=3,
            max_contexts_per_phase=3,
            max_model_initializations=1,
            max_model_requests=1,
            denoise_calls_per_request=35,
        ),
    )
    trial = SimpleNamespace(
        output=output,
        source_refs=lambda: sources,
        input_receipt=lambda _inputs: {"fixed": "input"},
        model_bundle=lambda _inputs: bundle,
    )
    registration = dict(
        phase="producer", inputs=trial.input_receipt(inputs), source_refs=sources, model_bundle_ref=bundle_ref
    )
    registration_ref = write(directory / "registration.json", registration)
    workers = []
    for name in ("simulation", "model"):
        log = directory / f"{name}.log"
        log.write_text("synthetic worker log", encoding="utf-8")
        workers.append(
            dict(
                name=name, actual_wait=True, reaped=True, returncode=0, signal_sent=False, log_ref=prefix.reference(log)
            )
        )
    lifecycle = dict(passed=True, launch_error=None, timed_out=False, interrupted=False, workers=workers)
    workers_ref = write(directory / "workers.json", lifecycle)
    counts = audit.summarize_branch(branch, "producer", inputs["config"])
    result = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        gpu_released=True,
        gpu_after=dict(memory_mib=0, compute_apps=[]),
        phase="producer",
        registration_ref=registration_ref,
        workers_ref=workers_ref,
        branch_ref=branch_ref,
        available_branch_ref=branch_ref,
        model_ref=model_ref,
        available_model_ref=model_ref,
        counts=counts,
        model_exchange=audit.verify_model_exchange(branch, model, bundle, TOOL),
    )
    prior_physical, prior_model = costs.zero_bounds()
    costs.phase_costs(
        trial,
        result,
        phase="producer",
        directory=directory,
        inputs=inputs,
        sources=sources,
        lifecycle=lifecycle,
        prior_physical=prior_physical,
        prior_model=prior_model,
    )
    return trial, inputs, result, sources, registration, lifecycle


def test_closed_producer_rechecks_real_historical_exchange_and_costs(tmp_path):
    trial, inputs, result, sources, _registration, _lifecycle = setup(tmp_path)
    physical, model = supervision.verify_closed_producer(trial, inputs, result, sources)
    assert physical["control_steps"] == dict(lower=16, upper=16)
    assert physical["native_steps"] == dict(lower=400, upper=400)
    assert model["requests_completed"] == dict(lower=1, upper=1)
    assert model["denoise_calls_completed"] == dict(lower=35, upper=35)


@pytest.mark.parametrize("change", ["gpu", "counts", "costs", "sources", "lifecycle", "logs", "model_ref"])
def test_changed_or_unreaped_producer_cannot_admit_checker(tmp_path, change):
    trial, inputs, result, sources, registration, lifecycle = setup(tmp_path)
    result = copy.deepcopy(result)
    directory = trial.output / "producer"
    if change == "gpu":
        result["gpu_released"] = False
    elif change == "counts":
        result["counts"]["control_steps"] -= 1
    elif change == "costs":
        result["cumulative_physical_bounds"]["native_steps"]["upper"] -= 1
    elif change == "sources":
        registration["source_refs"] = []
        result["registration_ref"] = write(directory / "registration.json", registration)
    elif change == "lifecycle":
        lifecycle["workers"][1]["reaped"] = False
        result["workers_ref"] = write(directory / "workers.json", lifecycle)
    elif change == "logs":
        (directory / "simulation.log").write_text("changed log", encoding="utf-8")
    else:
        result["model_ref"] = result["branch_ref"]
    with pytest.raises(ValueError):
        supervision.verify_closed_producer(trial, inputs, result, sources)
