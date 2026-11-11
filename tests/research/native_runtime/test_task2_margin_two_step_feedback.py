"""The feedback comparison binds evidence, budgets and independent replay."""

import sys
from types import ModuleType

import pytest

from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_guarded_two_step_feedback as previous
from scripts.simulation.runtime import run_task2_margin_two_step_feedback as runner
from tests.research.native_runtime.test_task2_earlier_restoration_feedback import branch


def test_worker_is_isolated_and_uses_margin_factory():
    worker = runner.isolated_worker()
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert worker.__globals__["ROOT"] == runner.ROOT and previous.ROOT != runner.ROOT
    assert worker.__globals__["full"].make_full_refiner is runner.make_margin_refiner


def test_matched_static_and_both_replayed_baselines_are_bound():
    inputs = runner.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [32, 80]
    receipt = runner.input_receipt(inputs)
    assert receipt["static_ref"] == runner.STATIC
    assert receipt["prior_producer_ref"] == runner.PRIOR_PRODUCER
    assert receipt["fallback_checker_ref"] == runner.FALLBACK_CHECKER


def test_margin_accounting_and_checker_import_isolation(monkeypatch):
    config = runner.read_config()[1]
    branches = [branch(10, 1), branch(11, 1)]
    for item in branches:
        item["branch"] = "margin_two_step"
        runtime = item["runtime"]
        runtime["quality_refiner"].update(quality_calls_completed=1, witness_audits_passed=1)
        runtime["first_hard_restoration_traces"] = [{"audit_status": "passed"}]
        runtime["feasible_start_traces"] = [
            dict(start_lp_attempts=1, solver_lp_attempted=False, seed_rollout_reused=True)
        ]
        runtime["two_step_feedback_traces"] = [dict(response_evaluations=4, phases=[], audit_status="passed")]
        runtime["margin_two_step_traces"] = [
            dict(
                model_position_margin_m=0.0005,
                model_rotation_margin_rad=0.003,
                selection_tolerance=1e-10,
                objective_scale=10000.0,
                original_acceptance_caps_changed=False,
                extra_model_responses=0,
                margin_prefix_evaluations=1,
                response_evaluations=4,
                audit_status="passed",
            )
        ]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["start_lp_calls"] == 2 and counts["margin_candidates_audited"] == 2
    branches[0]["runtime"]["margin_two_step_traces"][0]["model_position_margin_m"] = 0.0
    with pytest.raises(ValueError, match="Fixed margins"):
        runner.summarize_phase(branches, "producer", config)
    monkeypatch.setitem(sys.modules, runner.CHECKER_FORBIDDEN[-1], ModuleType(runner.CHECKER_FORBIDDEN[-1]))
    with pytest.raises(ValueError, match="candidates"):
        runner.replay_action([])
