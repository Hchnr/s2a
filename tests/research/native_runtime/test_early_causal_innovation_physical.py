"""One new action requires a completed innovation, independent gates, and a hard cap."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime import run_early_causal_innovation_physical as run


def period():
    return {
        "step": 121,
        "decision": {"quality": {"first_tcp_error": [0, 0, 0, 0, 0, 0], "verification": {"passed": True}}},
        "tcp_error": [0.002, 0, 0, 0, 0, 0],
        "task_success": True,
    }


def test_only_the_completed_innovation_enters_one_solver_call_and_then_stops():
    original = object()
    calls = []
    refiner = SimpleNamespace(solver=original, auditor=lambda *_args, **_kwargs: {})

    def solve(*_args, **kwargs):
        calls.append(kwargs)
        return {"witness_found": False, "status": "no_witness", "response_evaluations": 3}

    receipt = {}
    record = run.bind_after_completed(refiner, period(), receipt, solver=solve, expected_solver=original)
    first = refiner.solver("state", past_inner_error_rad="past")
    second = refiner.solver("state", past_inner_error_rad="past")
    assert first["status"] == "no_witness" and len(calls) == 1
    assert second == {"witness_found": False, "status": "registered_one_period_limit", "response_evaluations": 0}
    np.testing.assert_array_equal(calls[0]["translation_innovation"], [0.002, 0, 0])
    assert calls[0]["past_inner_error_rad"] == "past"
    assert record["last_completed_step"] == 121 and record["task_labels_used"] is False
    assert record["solver_calls"] == record["limit_stops"] == 1


def test_independent_innovation_gate_rejects_out_of_bounds_candidate():
    original = object()
    refiner = SimpleNamespace(
        solver=original,
        auditor=lambda *_args, **_kwargs: {
            "passed": True,
            "first_pose_caps_satisfied": True,
            "first_pose_caps_enforced": True,
            "first_tcp_error": np.array([0.009, 0, 0, 0, 0, 0]),
        },
    )
    record = run.bind_after_completed(refiner, period(), {}, solver=lambda *_a, **_k: None, expected_solver=original)
    witness = {"innovation_corrected_first_tcp_error": np.array([0.011, 0, 0, 0, 0, 0])}
    with pytest.raises(ValueError, match="innovation quality gates"):
        refiner.auditor(witness)
    assert record["independent_audit_attempts"] == 1 and record["independent_audit_passes"] == 0


def test_wrong_or_unverified_completed_period_cannot_bind():
    original = object()
    refiner = SimpleNamespace(solver=original, auditor=lambda *_args, **_kwargs: {})
    wrong = period()
    wrong["step"] = 122
    with pytest.raises(ValueError, match="completed step121"):
        run.bind_after_completed(refiner, wrong, {}, solver=lambda *_a, **_k: None, expected_solver=original)
    wrong = period()
    wrong["decision"]["quality"]["verification"]["passed"] = False
    with pytest.raises(ValueError, match="completed step121"):
        run.bind_after_completed(refiner, wrong, {}, solver=lambda *_a, **_k: None, expected_solver=original)


def test_native_runtime_receives_the_43_period_hard_budget(monkeypatch):
    seen = {}

    class Runtime:
        def _run(self, *_args, **kwargs):
            seen.update(kwargs)
            return "bounded"

    monkeypatch.setattr(run, "make_prefix_runtime", Runtime)
    names = ("runtime_config", "record", "record_xml", "commands", "saved", "receipts", "asset_refs", "controller")
    old = dict.fromkeys(names, object())
    assert run.bounded_prefix(old, Path("outputs/example"), lambda *_a, **_k: None) == "bounded"
    assert seen["continuation_budget"] == 43 and seen["continuation_limit"] == 48
    assert seen["registered_seed"] == 42 and seen["capture_final_prefix_inner"] is True


def test_checker_import_does_not_load_any_candidate_solver():
    code = """
from scripts.simulation.runtime import run_early_causal_innovation_physical as run
run.checker_isolation()
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
    )
    assert child.returncode == 0, child.stdout + child.stderr
