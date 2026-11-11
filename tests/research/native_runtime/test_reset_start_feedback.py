"""Fixed reset feedback budgets, safe stopping, and checker isolation."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import run_reset_start_feedback as study


def branch(index, phase="producer", count=0):
    period = {"pose_caps_satisfied": True, "inner_below_buffer_samples": 0}
    runtime = {
        "reset_observation_verified": True,
        "reset_image_match": True,
        "cleanup_complete": True,
        "previous_saved_command": None,
        "canvas_thread_scope": {"before": 1, "requested": 2, "inside": 2, "after": 1},
        "control_steps_completed": count,
        "newly_selected_commands": count if phase == "producer" else 0,
        "counters": {
            "model_construct": {"completed": 3},
            "context": {"completed": 3},
            "mj_step": {"attempted": 25 * count, "completed": 25 * count},
            "mj_step2": {"attempted": 0},
        },
        "continuation": {"periods": [copy.deepcopy(period) for _ in range(count)], "stop": None},
    }
    return {
        "complete": True,
        "passed": True,
        "phase": phase,
        "case": {"case_index": index},
        "candidate_modules_unloaded": phase == "checker",
        "runtime": runtime,
    }


def test_fixed_budget_rejects_more_steps_and_implicit_retries():
    config = json.loads(study.CONFIG.read_text())
    study.validate_config(config)
    for key, value in (
        ("continuation_steps", 9),
        ("max_native_steps_per_phase", 2401),
        ("world_model_calls", False),
        ("automatic_retry", True),
        ("case_indices", [0]),
    ):
        with pytest.raises(ValueError):
            study.validate_config(dict(config, **{key: value}))


def test_summary_requires_all_cases_and_counts_only_complete_safe_windows():
    config = json.loads(study.CONFIG.read_text())
    cases = [branch(i, count=8 if i == 0 else 0) for i in range(12)]
    result = study.summarize_phase(cases, "producer", config)
    assert result["totals"]["short_window_passes"] == 1 and result["totals"]["control_steps"] == 8
    assert result["totals"]["native_steps"] == 200 and result["totals"]["models"] == 36
    cases[0]["runtime"]["continuation"]["periods"][3]["inner_below_buffer_samples"] = 1
    assert study.summarize_phase(cases, "producer", config)["totals"]["short_window_passes"] == 0
    cases[0]["runtime"]["counters"]["mj_step"]["attempted"] = 201
    with pytest.raises(ValueError, match="physical budgets"):
        study.summarize_phase(cases, "producer", config)


def test_checker_preserves_unrecomputed_solver_stop_without_fabricating_physics():
    outcome = {"periods": [], "stop": None}
    stop = {"reason": "quality_no_witness", "at_step": 0, "next_action_dispatched": False}
    checked = study.reconcile_stop(outcome, stop, current_margin=0.4)
    assert checked["stop"] == stop and checked["decision_stop_recorded_not_recomputed"] is True
    stop = {
        "reason": "observed_joint_buffer_loss",
        "at_step": 0,
        "next_action_dispatched": False,
        "minimum_current_margin_rad": 0.04,
    }
    checked = study.reconcile_stop({"periods": [], "stop": None}, stop, current_margin=0.04)
    assert checked["decision_stop_recorded_not_recomputed"] is False
    with pytest.raises(ValueError, match="observable"):
        study.reconcile_stop({"periods": [], "stop": None}, stop, current_margin=0.4)


def test_checker_import_does_not_load_candidate_or_simulator():
    code = (
        "from scripts.simulation.runtime import run_reset_start_feedback as s; "
        "s.checker_isolation(); s.require_no_simulator_import()"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
