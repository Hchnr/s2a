"""Verify binding-free historical inputs and conservative failure accounting."""

import copy
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import adaptive_recovery_inputs as recovery


def test_recorded_inputs_match_the_historical_loader_without_binding_globals():
    script = """\
import sys
import numpy as np
from scripts.simulation.runtime import adaptive_live_bowl_admission as historical
from scripts.simulation.runtime import adaptive_recovery_inputs as recovery
from scripts.simulation.runtime import registered_model_bundle as model_admission
from scripts.simulation.runtime import run_causal_replan_step107_model as model

old = model.previous
older = old.previous
before = (old.CONFIG, old.PLAN, old.OUTPUT, old.source_refs,
          older.CONFIG, older.PLAN, older.OUTPUT, older.load_inputs, older.source_refs)
sources_before = model_admission.prior_adaptive_source_refs()
actual = recovery.load_recorded_inputs()
after = (old.CONFIG, old.PLAN, old.OUTPUT, old.source_refs,
         older.CONFIG, older.PLAN, older.OUTPUT, older.load_inputs, older.source_refs)
assert before == after
assert sources_before == model_admission.prior_adaptive_source_refs()
expected = historical.load_inputs()
np.testing.assert_equal(actual, expected)
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
    subprocess.run([sys.executable, "-c", script], cwd=recovery.prefix.REPO, env=env, check=True, timeout=900)


def test_actual_failed_attempt_preserves_unknown_physical_cost_and_known_model_cost():
    config = recovery.prefix.read(recovery.prefix.reference(recovery.historical.CONFIG))
    result = recovery.load_failed_attempt(config)
    assert result["simulation_bounds"]["control_steps"] == dict(lower=0, upper=115, evidence="not_recorded")
    assert result["simulation_bounds"]["native_steps"] == dict(lower=0, upper=2875, evidence="not_recorded")
    assert result["cumulative_physical_bounds"] == dict(
        control_steps=dict(lower=1531, upper=1646),
        native_steps=dict(lower=38275, upper=41150),
        models=dict(lower=78, upper=81),
        contexts=dict(lower=78, upper=81),
    )
    assert result["previous_model_initializations"] == 9 and result["previous_requests_attempted"] == 21


@pytest.mark.parametrize("change", ["gpu", "gpu_uuid", "model_attempt", "signal", "timeout", "budget"])
def test_changed_failure_evidence_or_budget_is_rejected(monkeypatch, change):
    read = recovery.prefix.read
    config = read(recovery.prefix.reference(recovery.historical.CONFIG))
    failure = read(recovery.FAILURE)
    workers = read(failure["workers_ref"])
    model = read(failure["available_model_ref"])
    config, failure, workers, model = copy.deepcopy((config, failure, workers, model))
    if change == "gpu":
        failure["gpu_released"] = False
    elif change == "gpu_uuid":
        model["gpu_uuid"] = "GPU-other"
    elif change == "model_attempt":
        failure["model_requests_attempted"] = model["model_requests_attempted"] = 1
    elif change == "signal":
        next(row for row in workers["workers"] if row["name"] == "simulation")["signal_sent"] = False
    elif change == "timeout":
        workers["timed_out"] = True
    else:
        config["max_control_steps_per_phase"] += 1

    def replacement(ref):
        if ref == recovery.FAILURE:
            return failure
        if ref == failure["workers_ref"]:
            return workers
        if ref == failure["available_model_ref"]:
            return model
        return read(ref)

    monkeypatch.setattr(recovery.prefix, "read", replacement)
    with pytest.raises(ValueError):
        recovery.load_failed_attempt(config)
