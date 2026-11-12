"""The next-reference phase preserves its worker and physical budget contracts."""

import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as first_replay
from scripts.simulation.runtime import run_reset_bowl_next_reference as owner
from tests.research.native_runtime.test_reset_bowl_next_reference import run_next, setup_next


@pytest.fixture(autouse=True)
def synthetic_isolation(monkeypatch):
    monkeypatch.setattr(first_replay.isolation, "checker_isolation", lambda: None)
    monkeypatch.setattr(first_replay.original, "checker_isolation", lambda: None)


@pytest.fixture
def produced_branch(tmp_path):
    case, receipt, options, _, _ = setup_next(tmp_path)
    continuation = run_next(case, receipt, options)
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
            "mj_step": {"attempted": 3200, "completed": 3200},
            "model_construct": {"completed": 3},
            "context": {"completed": 3},
        },
    }
    return {"complete": True, "passed": True, "runtime": runtime}


def test_counts_actual_prefix_reference_and_physics(produced_branch):
    counts = owner.summarize_branch(produced_branch, "producer", owner.FIXED)
    assert counts["control_steps"] == 128 and counts["native_steps"] == 3200
    assert counts["prefix_steps"] == 88 and counts["new_reference_steps"] == 40
    assert counts["models"] == counts["contexts"] == 3


def test_rejects_unaccounted_physics_or_candidate_work(produced_branch):
    produced_branch["runtime"]["counters"]["mj_step"]["completed"] -= 1
    with pytest.raises(ValueError, match="actual periods"):
        owner.summarize_branch(produced_branch, "producer", owner.FIXED)
    produced_branch["runtime"]["counters"]["mj_step"]["completed"] += 1
    produced_branch["runtime"]["segment_receipts"][0]["quality_refiner"]["endpoint_queries_charged"] = 25601
    with pytest.raises(ValueError, match="candidate budgets"):
        owner.summarize_branch(produced_branch, "producer", owner.FIXED)


def test_worker_adapter_extends_only_the_registered_bound(monkeypatch):
    from s2a.research.native_runtime import reset_bounded_runtime as bounded
    from scripts.simulation.runtime import reset_bowl_next_reference as next_reference
    from scripts.simulation.runtime import reset_bowl_reference_tail as old_tail

    for name in ("OUTPUT", "load_inputs", "input_receipt", "source_refs"):
        monkeypatch.setattr(owner.previous, name, getattr(owner.previous, name))
    for name in ("run_producer", "run_checker"):
        monkeypatch.setattr(old_tail, name, getattr(old_tail, name))
    base = bounded.ResetBoundedRuntime
    monkeypatch.setattr(base, "run_feedback", lambda _self, *args, max_control_steps, **kwargs: max_control_steps)
    monkeypatch.setattr(bounded, "ResetBoundedRuntime", base)
    captured = []
    monkeypatch.setattr(next_reference, "run_producer", lambda *args, **kwargs: captured.append(kwargs))
    inputs = dict(previous_88_producer="prior", capture88="capture", reference88="raw", envelope88="envelope")
    owner.bind_worker(inputs)
    adapter = bounded.ResetBoundedRuntime.__new__(bounded.ResetBoundedRuntime)
    assert adapter.run_feedback(max_control_steps=88) == 128
    with pytest.raises(ValueError, match="original bound"):
        adapter.run_feedback(max_control_steps=56)
    old_tail.run_producer()
    assert captured[0]["previous_88_producer"] == "prior"
    assert captured[0]["load_step88_reference"]() == ("raw", "envelope")


def test_checker_import_does_not_load_candidate_or_simulator():
    code = (
        "from scripts.simulation.runtime import run_reset_bowl_next_reference as owner; "
        "owner.previous.isolation.checker_isolation(); owner.previous.original.require_no_simulator_import()"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
