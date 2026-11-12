"""Check recovery identity, prior uncertainty and phase-local registration."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_recovery_admission as admission


def configs():
    return json.loads(admission.CONFIG.read_text()), json.loads(admission.historical.CONFIG.read_text())


def test_exact_registered_config_and_distinct_session_are_accepted():
    config, old = configs()
    admission.validate_config(config, old)
    assert config["session"] == admission.SESSION != admission.historical.SESSION


@pytest.mark.parametrize(
    "change", ["seed", "session", "gpu_uuid", "budget", "uncertainty", "output", "failed_sha", "type", "extra"]
)
def test_changed_recovery_identity_or_budget_is_rejected(change):
    config, old = configs()
    if change == "seed":
        config["seed"] = 43
    elif change == "session":
        config["session"] = admission.historical.SESSION
    elif change == "gpu_uuid":
        config["gpu_uuid"] = "GPU-other"
    elif change == "budget":
        config["max_model_requests"] += 1
    elif change == "uncertainty":
        config["unknown_prior_control_steps_upper"] = 0
    elif change == "output":
        config["output"] = str(admission.historical.OUTPUT.relative_to(admission.ROOT))
    elif change == "failed_sha":
        config["prior_failed_attempt_sha256"] = "f" * 64
    elif change == "type":
        config["max_model_initializations"] = True
    else:
        config["unexpected"] = True
    with pytest.raises(ValueError, match="recovery identity, prior uncertainty and budgets"):
        admission.validate_config(config, old)


def test_registration_is_unique_and_limited_to_recovery_phase_roots():
    config, _old = configs()
    inputs = dict(config=config, initial=dict(step=107, sim_time_s=5.35, state_sha256="a" * 64))
    registration = admission.model_registration(inputs, admission.OUTPUT / "producer/simulation")
    assert registration["session"] == admission.SESSION
    assert registration["max_requests"] == 2 and registration["max_periods"] == 8
    assert registration["image_root"] == "outputs/adaptive_live_bowl_recovery_20261112/producer/simulation"
    checker = admission.model_registration(inputs, admission.OUTPUT / "checker/simulation")
    assert checker["image_root"] == "outputs/adaptive_live_bowl_recovery_20261112/checker/simulation"
    with pytest.raises(ValueError, match="phase directory"):
        admission.model_registration(inputs, admission.historical.OUTPUT / "producer/simulation")
    inputs = copy.deepcopy(inputs)
    inputs["config"]["session"] = admission.historical.SESSION
    with pytest.raises(ValueError, match="phase directory"):
        admission.model_registration(inputs, admission.OUTPUT / "producer/simulation")


@pytest.mark.parametrize("kind", ["config", "plan"])
def test_changed_registration_file_is_rejected_before_loading_historical_inputs(tmp_path, monkeypatch, kind):
    path = tmp_path / f"changed_{kind}"
    path.write_text("changed")
    monkeypatch.setattr(admission, kind.upper(), path)
    monkeypatch.setattr(admission.recovery, "load_recorded_inputs", lambda: pytest.fail("historical inputs accessed"))
    with pytest.raises(ValueError, match="Changed input"):
        admission.load_inputs()


def test_fresh_process_loads_actual_inputs_without_legacy_binding_or_gpu():
    script = """\
import sys
from scripts.simulation.runtime import adaptive_live_bowl_recovery_admission as admission
from scripts.simulation.runtime import registered_model_bundle as manifest
from scripts.simulation.runtime import run_causal_replan_step107_model as model

old, older = model.previous, model.previous.previous
before = (old.CONFIG, old.PLAN, old.OUTPUT, old.source_refs,
          older.CONFIG, older.PLAN, older.OUTPUT, older.load_inputs, older.source_refs)
sources = manifest.prior_adaptive_source_refs()
inputs = admission.load_inputs()
after = (old.CONFIG, old.PLAN, old.OUTPUT, old.source_refs,
         older.CONFIG, older.PLAN, older.OUTPUT, older.load_inputs, older.source_refs)
assert before == after and sources == manifest.prior_adaptive_source_refs()
assert inputs['config']['session'] == admission.SESSION
assert inputs['failed_attempt']['cumulative_physical_bounds']['control_steps'] == {'lower': 1531, 'upper': 1646}
assert inputs['failed_attempt']['cumulative_physical_bounds']['native_steps'] == {'lower': 38275, 'upper': 41150}
assert inputs['refs']['failed_attempt']['sha256'] == admission.recovery.FAILURE['sha256']
registration = admission.model_registration(inputs, admission.OUTPUT / 'producer/simulation')
assert registration['initial'] == inputs['initial'] and registration['max_requests'] == 2
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
    subprocess.run([sys.executable, "-c", script], cwd=admission.ROOT, env=env, check=True, timeout=900)
