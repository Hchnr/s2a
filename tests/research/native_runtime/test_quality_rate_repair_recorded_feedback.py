"""Physical stream injection, saved relay, and checker isolation remain bounded."""

import copy
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as admission
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as run


def test_solver_injection_changes_only_the_selected_refiner():
    from s2a.research.trajectory_search import quality_capped_inner_joint_plan, quality_rate_repair_solver

    original = quality_capped_inner_joint_plan.solve_interior_plan
    refiner = SimpleNamespace(solver=original)
    policy = SimpleNamespace(arm_policy=SimpleNamespace(tracking=SimpleNamespace(_refiner=refiner)))
    receipt = {}
    run.inject_solver(policy, receipt)
    assert refiner.solver is quality_rate_repair_solver.solve_interior_plan
    assert quality_capped_inner_joint_plan.solve_interior_plan is original
    assert receipt["quality_rate_repair_solver"]["injected"] is True
    with pytest.raises(ValueError, match="original quality solver"):
        run.inject_solver(policy, {})


def test_saved_relay_accepts_a_bounded_stop_without_a_model_request():
    inputs = admission.load_inputs()

    class StopChannel:
        def receive(self):
            return {"version": 1, "kind": "stop", "session": "recorded_correct08_seed42_20261111", "next_step": 80}

        def send(self, _value):
            pytest.fail("a stopped relay must not send a response")

    result = {"local_reference_requests": 0, "responses_sent": 0, "records": []}
    run.serve_saved(StopChannel(), inputs, result, image_root=run.OUTPUT / "producer/simulation/runtime")
    assert result["stop_frame"]["next_step"] == 80
    assert result["local_reference_requests"] == result["responses_sent"] == 0


def test_saved_relay_rejects_a_changed_historical_denoise_count():
    inputs = admission.load_inputs()
    inputs["records"] = copy.deepcopy(inputs["records"])
    inputs["records"][3]["historical_denoise_calls"] = 34
    with pytest.raises(ValueError, match="saved response"):
        run.relay_service(inputs, run.OUTPUT / "producer/simulation/runtime")


def test_checker_import_stays_independent_of_the_repaired_solver():
    code = """
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as run
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


def test_development_success_metrics_do_not_claim_control_qualification():
    rows = [
        {
            "task_success": success,
            "pose_caps_satisfied": True,
            "minimum_inner_margin_rad": 0.5,
            "inner_below_buffer_samples": 0,
            "inner_outside_samples": 0,
        }
        for success in ([False] + [True] * 40)
    ]
    branch = {"runtime": {"continuation": {"segments": [{"continuation": {"periods": rows}}]}}}
    result = run.metrics(branch)
    assert result["terminal_consecutive_success_periods"] == 40
    assert result["control_qualified"] is False
