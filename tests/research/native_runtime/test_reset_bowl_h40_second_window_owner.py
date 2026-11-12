"""The second-window owner accounts for actual physical work and isolation."""

import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as first_replay
from scripts.simulation.runtime import run_reset_bowl_h40_second_window as owner
from tests.research.native_runtime.test_reset_bowl_h40_second_window import produce, setup_second


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    monkeypatch.setattr(first_replay.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(first_replay.original, "checker_isolation", lambda: None)


@pytest.fixture
def produced_branch(tmp_path):
    case, receipt, options, _, _ = setup_second(tmp_path)
    continuation = produce(case, receipt, options)
    candidate = dict(
        quality_refiner=dict(
            quality_response_evaluations=0,
            baseline_response_evaluations=0,
            endpoint_queries_charged=0,
            transition_steps_charged=0,
        ),
        gripper_proposals=dict(candidate_sequences=0),
    )
    runtime = {
        **receipt,
        "continuation": continuation,
        "complete": True,
        "passed": True,
        "cleanup_complete": True,
        "reset_observation_verified": True,
        "reset_image_match": True,
        "previous_saved_command": None,
        "segment_receipts": [candidate],
        "counters": {
            "mj_step": {"attempted": 1400, "completed": 1400},
            "model_construct": {"completed": 3},
            "context": {"completed": 3},
        },
    }
    return {"complete": True, "passed": True, "runtime": runtime}


def test_counts_only_dispatched_periods_and_physics(produced_branch):
    counts = owner.summarize_branch(produced_branch, "producer")
    assert counts["control_steps"] == 56 and counts["native_steps"] == 1400
    assert counts["second_reference_steps"] == 8 and counts["models"] == counts["contexts"] == 3


def test_rejects_a_mismatched_inner_step_count(produced_branch):
    produced_branch["runtime"]["counters"]["mj_step"]["completed"] -= 1
    with pytest.raises(ValueError, match="actual periods"):
        owner.summarize_branch(produced_branch, "producer")


def test_rejects_candidate_work_above_the_registered_budget(produced_branch):
    produced_branch["runtime"]["segment_receipts"][0]["quality_refiner"]["endpoint_queries_charged"] = 11201
    with pytest.raises(ValueError, match="candidate budgets"):
        owner.summarize_branch(produced_branch, "producer")


def test_checker_import_does_not_load_candidate_or_simulator():
    code = (
        "from scripts.simulation.runtime import run_reset_bowl_h40_second_window as owner; "
        "owner.isolation.checker_isolation(); owner.original.require_no_simulator_import()"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
