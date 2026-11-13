"""Account for completed, interrupted and missing trial receipts."""

import json
from types import SimpleNamespace

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_recovery_owner as accounting
from scripts.simulation.runtime import reset_seed43_costs as costs


def context(tmp_path):
    directory = tmp_path / "producer"
    directory.mkdir()
    sources = [{"source": "registered"}]
    trial = SimpleNamespace(source_refs=lambda: sources)
    config = dict(
        max_control_steps_per_phase=16,
        max_native_steps_per_phase=400,
        max_models_per_phase=3,
        max_contexts_per_phase=3,
        max_model_initializations=1,
        max_model_requests=1,
        denoise_calls_per_request=35,
    )
    return directory, sources, trial, {"config": config}


def execute(items, *, phase="producer", lifecycle=None, result=None):
    directory, sources, trial, inputs = items
    result = {"phase": phase} if result is None else result
    physical, model = costs.zero_bounds()
    costs.phase_costs(
        trial,
        result,
        phase=phase,
        directory=directory,
        inputs=inputs,
        sources=sources,
        lifecycle=lifecycle,
        prior_physical=physical,
        prior_model=model,
    )
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def receipts(directory, *, attempted=8, completed=8, denoises=35):
    runtime = dict(
        control_steps_attempted=attempted,
        control_steps_completed=completed,
        counters=dict(
            mj_step=dict(attempted=25 * attempted, completed=25 * completed),
            model_construct=dict(attempted=3, completed=3),
            context=dict(attempted=3, completed=3),
        ),
    )
    write(directory / "simulation/result.json", {"runtime": runtime})
    write(
        directory / "model/result.json",
        dict(
            model_initialization_attempts=1,
            model_initializations=1,
            model_requests_attempted=1,
            model_requests_completed=1 if denoises == 35 else 0,
            denoise_calls_attempted=denoises,
            denoise_calls_completed=denoises,
        ),
    )
    return {"workers": [dict(name=name, actual_wait=True, reaped=True) for name in ("simulation", "model")]}


def test_missing_or_unreaped_receipts_remain_bounded_unknowns(tmp_path):
    items = context(tmp_path)
    lifecycle = receipts(items[0])
    for state in (None, {"workers": [dict(name="simulation", actual_wait=False, reaped=False)]}):
        result = execute(items, lifecycle=state)
        assert result["phase_physical_bounds"]["native_steps"] == dict(lower=0, upper=400, evidence="not_recorded")
        assert result["phase_model_bounds"]["denoise_calls_completed"] == dict(
            lower=0, upper=35, evidence="not_recorded"
        )
    exact = execute(items, lifecycle=lifecycle)
    assert exact["phase_physical_bounds"]["native_steps"] == dict(lower=200, upper=200, evidence="final_receipt")
    assert exact["phase_model_bounds"]["denoise_calls_completed"] == dict(lower=35, upper=35, evidence="final_receipt")
    exact.update(counts=dict(control_steps=8, native_steps=200, models=3, contexts=3))
    accounting.require_complete_phase_costs(exact)


def test_partial_final_receipt_preserves_attempted_and_completed_bounds(tmp_path):
    items = context(tmp_path)
    lifecycle = receipts(items[0], attempted=3, completed=2, denoises=3)
    result = execute(items, lifecycle=lifecycle)
    assert result["phase_physical_bounds"]["control_steps"] == dict(lower=2, upper=3, evidence="final_receipt")
    assert result["phase_physical_bounds"]["native_steps"] == dict(lower=50, upper=75, evidence="final_receipt")
    assert result["phase_model_bounds"]["requests_completed"] == dict(lower=0, upper=1, evidence="final_receipt")
    assert result["phase_model_bounds"]["denoise_calls_attempted"] == dict(lower=3, upper=3, evidence="final_receipt")


def test_bad_counters_or_changed_sources_never_collapse_unknowns(tmp_path):
    items = context(tmp_path)
    lifecycle = receipts(items[0], attempted=17, completed=17)
    result = execute(items, lifecycle=lifecycle)
    assert result["phase_physical_bounds"]["control_steps"]["upper"] == 16
    assert "physical_receipt_error" in result
    items[2].source_refs = lambda: [{"source": "changed"}]
    result = execute(items, lifecycle=lifecycle)
    assert not result["partial_source_match"]
    assert result["phase_model_bounds"]["requests_completed"]["upper"] == 1


def test_checker_never_claims_model_work(tmp_path):
    items = context(tmp_path)
    result = execute(items, phase="checker")
    assert all(row["lower"] == row["upper"] == 0 for row in result["phase_model_bounds"].values())
    assert result["phase_physical_bounds"]["control_steps"]["upper"] == 16


def test_checker_costs_accumulate_with_exact_producer_costs(tmp_path):
    directory, sources, trial, inputs = context(tmp_path)
    receipts(directory)
    lifecycle = {"workers": [dict(key="checker", actual_wait=True, reaped=True)]}
    prior_physical = {
        name: dict(lower=value, upper=value)
        for name, value in dict(control_steps=8, native_steps=200, models=3, contexts=3).items()
    }
    prior_model = {
        name: dict(lower=value, upper=value)
        for name, value in dict(
            initializations_completed=1, requests_attempted=1, requests_completed=1, denoise_calls_completed=35
        ).items()
    }
    result = {"phase": "checker", "counts": dict(control_steps=8, native_steps=200, models=3, contexts=3)}
    costs.phase_costs(
        trial,
        result,
        phase="checker",
        directory=directory,
        inputs=inputs,
        sources=sources,
        lifecycle=lifecycle,
        prior_physical=prior_physical,
        prior_model=prior_model,
    )
    accounting.require_complete_phase_costs(result)
    assert result["cumulative_physical_bounds"]["native_steps"] == dict(lower=400, upper=400)
    assert result["cumulative_physical_bounds"]["models"] == dict(lower=6, upper=6)
    assert result["cumulative_model_bounds"]["denoise_calls_completed"] == dict(lower=35, upper=35)


@pytest.mark.parametrize(
    "change", [dict(max_control_steps_per_phase=17), dict(max_model_requests=2), dict(denoise_calls_per_request=True)]
)
def test_changed_budgets_are_refused(tmp_path, change):
    items = context(tmp_path)
    items[3]["config"].update(change)
    with pytest.raises(ValueError, match="fixed two-window"):
        execute(items)
