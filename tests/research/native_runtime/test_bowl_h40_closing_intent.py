"""Verify the modified reference in a synthetic producer and isolated checker."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from s2a.references.opening_intent import amplified_opening_h40
from scripts.simulation.runtime import reset_bowl_fixed_h40 as fixed
from scripts.simulation.runtime import run_bowl_h40_closing_intent as intent
from tests.research.native_runtime.test_reset_bowl_fixed_h40 import setup_fixed
from tests.research.native_runtime.test_reset_live_two_window import terminal_at


def test_config_freezes_causal_transform_and_actual_prior_cost(monkeypatch):
    config = json.loads(intent.CONFIG.read_text())
    monkeypatch.setattr(intent.admission, "FIXED", intent.FIXED)
    intent.admission.validate_config(config)
    for key, value in (
        ("opening_lead_steps", 9),
        ("opening_gain", 2.0),
        ("before_probe_periods", 0),
        ("new_model_requests", 1),
    ):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="fixed bowl case"):
            intent.admission.validate_config(changed)


def test_admission_and_checker_do_not_load_candidate_or_simulator():
    code = (
        "import sys; from scripts.simulation.runtime import run_bowl_h40_closing_intent as s; "
        "e=s.bind('checker'); x=e.admission.load_inputs(verify_assets=False); "
        "assert x['refs']['opening_lead_audit']==s.LEAD_AUDIT and x['refs']['static_close_audit']==s.STATIC_AUDIT; "
        "assert x['config']['opening_gain']==3.0 and x['config']['opening_lead_steps']==8; "
        "assert 's2a.references.opening_intent' not in sys.modules and 'mujoco' not in sys.modules; "
        "e.isolation.checker_isolation()"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_synthetic_checker_replays_modified_reference_and_rejects_original(monkeypatch, tmp_path):
    from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as checker_module
    from tests.research.native_runtime.test_reset_live_two_window import score
    from tests.research.native_runtime.test_streaming_continuation import fake_case

    produced, receipt, args, _ = setup_fixed(tmp_path)
    old_load, saved = args["load_reference"], []

    def varied():
        raw, envelope = copy.deepcopy(old_load())
        action = np.asarray(envelope["action"], dtype=float)
        action[:, 9] = np.linspace(1, 0.5, len(action))
        envelope["action"] = action.tolist()
        raw["outputs"][0]["content"]["action"] = action.tolist()
        saved.append((raw, envelope))
        return raw, envelope

    args["load_reference"] = varied
    monkeypatch.setattr(
        fixed,
        "remaining_h40",
        lambda poses, openings, *, consumed: amplified_opening_h40(
            poses, openings, consumed=consumed, lead_steps=8, gain=3
        ),
    )
    run = fixed.run_producer(produced.env, terminal_at(produced, 0), produced.observers, receipt, **args)
    producer = {"runtime": {**receipt, "continuation": run}}
    assert [s["opening_terminal_hold_endpoints"] for s in run["fixed"]["segments"]] == [8, 16, 24, 32, 40]
    monkeypatch.setattr(checker_module.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(checker_module.original, "checker_isolation", lambda: None)

    def replay(name):
        directory = tmp_path / name
        directory.mkdir()
        checker = fake_case(directory, windows=5)
        checker.observers.render_anchor = produced.observers.render_anchor
        checked = dict(control_steps_attempted=0, control_steps_completed=0, newly_selected_commands=0)
        output = checker_module.run_checker(
            checker.env,
            terminal_at(checker, 0),
            checker.observers,
            checked,
            case=args["case"],
            producer=producer,
            expected_capture=args["expected_capture"],
            load_reference=lambda: saved[0],
            runtime_output=directory,
            state_reader=checker.state_reader,
            task_observer=checker.task_observer,
            gripper_score=score,
            render_scope=lambda render, _scope, *a, **kw: render(*a, **kw),
        )
        return checker, checked, output

    monkeypatch.setattr(checker_module, "remaining_h40", intent.checker_window)
    checker, checked, result = replay("checker_intent")
    assert checker.sim.inner == 1200 and checked["newly_selected_commands"] == 0
    assert [r["steps_completed"] for r in result["replays"]] == [8] * 5
    monkeypatch.setattr(checker_module, "remaining_h40", intent.remaining_h40)
    with pytest.raises((AssertionError, ValueError)):
        replay("checker_original")
