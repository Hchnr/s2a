"""Bind the candidate and independent checker to the recorded development input."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from s2a.references.opening_lead import opening_lead_h40
from scripts.simulation.runtime import reset_bowl_fixed_h40 as fixed
from scripts.simulation.runtime import run_bowl_h40_opening_lead as lead
from tests.research.native_runtime.test_reset_bowl_fixed_h40 import setup_fixed
from tests.research.native_runtime.test_reset_live_two_window import terminal_at


def test_config_freezes_the_single_lead_and_prior_resource_budget(monkeypatch):
    config = json.loads(lead.CONFIG.read_text())
    monkeypatch.setattr(lead.admission, "FIXED", lead.FIXED)
    lead.admission.validate_config(config)
    for key, value in (
        ("opening_lead_steps", 5),
        ("before_probe_periods", 0),
        ("max_native_steps_per_phase", 1201),
        ("new_model_requests", 1),
    ):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match="fixed bowl case"):
            lead.admission.validate_config(changed)


def test_real_admission_and_checker_do_not_load_candidate_or_simulator():
    code = (
        "import sys; from scripts.simulation.runtime import run_bowl_h40_opening_lead as s; "
        "e=s.bind('checker'); x=e.admission.load_inputs(verify_assets=False); "
        "assert x['refs']['fixed_h40_audit'] == s.BASELINE; "
        "assert x['refs']['saved_contacts_audit'] == s.CONTACTS; "
        "assert x['config']['opening_lead_steps'] == 4; "
        "assert 's2a.references.opening_lead' not in sys.modules and 'mujoco' not in sys.modules; "
        "e.isolation.checker_isolation()"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_lead_is_applied_to_all_five_real_runtime_segments(monkeypatch, tmp_path):
    case, receipt, args, _ = setup_fixed(tmp_path)
    calls = []

    def shifted(poses, openings, *, consumed):
        result = opening_lead_h40(poses, openings, consumed=consumed, lead_steps=4)
        calls.append((consumed, np.asarray(openings).copy(), result[1].copy()))
        return result

    monkeypatch.setattr(fixed, "remaining_h40", shifted)
    result = fixed.run_producer(case.env, terminal_at(case, 0), case.observers, receipt, **args)
    assert result["fixed"]["complete"] and case.sim.inner == 1200
    assert [consumed for consumed, _, _ in calls] == [0, 8, 16, 24, 32]
    for consumed, source, actual in calls:
        np.testing.assert_array_equal(actual, source[np.minimum(consumed + np.arange(40) + 4, 39)])
    assert [row["opening_terminal_hold_endpoints"] for row in result["fixed"]["segments"]] == [4, 12, 20, 28, 36]


def test_checker_replays_shifted_scores_and_rejects_unshifted_reference(monkeypatch, tmp_path):
    from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as checker_module
    from tests.research.native_runtime.test_reset_live_two_window import score
    from tests.research.native_runtime.test_streaming_continuation import fake_case

    produced, receipt, args, _ = setup_fixed(tmp_path)
    original_load = args["load_reference"]
    saved = []

    def load_varied_reference():
        raw, envelope = copy.deepcopy(original_load())
        action = np.asarray(envelope["action"], dtype=float)
        action[:, 9] = np.linspace(1, 0, len(action))
        envelope["action"] = action.tolist()
        raw["outputs"][0]["content"]["action"] = action.tolist()
        saved.append((raw, envelope))
        return raw, envelope

    args["load_reference"] = load_varied_reference
    monkeypatch.setattr(
        fixed,
        "remaining_h40",
        lambda poses, openings, *, consumed: opening_lead_h40(poses, openings, consumed=consumed, lead_steps=4),
    )
    result = fixed.run_producer(produced.env, terminal_at(produced, 0), produced.observers, receipt, **args)
    producer = {"runtime": {**receipt, "continuation": result}}
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

    monkeypatch.setattr(checker_module, "remaining_h40", lead.checker_window)
    checker, checked, output = replay("checker_lead")
    assert checker.sim.inner == 1200 and checked["newly_selected_commands"] == 0
    assert [window["steps_completed"] for window in output["replays"]] == [8] * 5
    monkeypatch.setattr(checker_module, "remaining_h40", lead.remaining_h40)
    with pytest.raises((AssertionError, ValueError)):
        replay("checker_wrong")
