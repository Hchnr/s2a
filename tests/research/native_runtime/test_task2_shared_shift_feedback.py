"""Shared-shift feedback binds provenance, budgets and isolated replay."""

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from scripts.simulation.runtime import run_task2_shared_shift_feedback as runner
from tests.research.native_runtime.test_task2_earlier_restoration_feedback import branch


def test_isolated_worker_and_matched_prior_inputs():
    worker = runner.isolated_worker()
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert worker.__globals__["ROOT"] == runner.ROOT
    assert worker.__globals__["full"].make_full_refiner is runner.make_shared_shift_refiner
    inputs = runner.load_inputs()
    assert [case["spec"]["prefix_steps"] for case in inputs["cases"]] == [32, 80]
    assert runner.input_receipt(inputs)["static_ref"] == runner.STATIC
    for case in inputs["cases"]:
        policy_case = runner.original_binding.policy_case(case)
        assert "feedback_periods" not in policy_case and "stop_inputs" not in policy_case


def test_factory_attaches_the_actual_transaction_history(monkeypatch):
    from s2a.research.trajectory_search.adaptive_plan_history import AdaptivePlanHistory

    history = AdaptivePlanHistory()
    policy = SimpleNamespace(tracking=SimpleNamespace(history=history))
    captured = {}

    def build(_env, _case, receipt, factory):
        captured["adapter"] = factory(receipt)
        return policy, "inner_history"

    monkeypatch.setattr(runner.builder, "build_refiner", build)
    assert runner.make_shared_shift_refiner(None, None, {}) == (policy, "inner_history")
    assert captured["adapter"]._history is history


def test_shared_costs_include_shift_and_checker_isolation(monkeypatch):
    config = runner.read_config()[1]
    branches = [branch(10, 1), branch(11, 1)]
    for item in branches:
        item["branch"] = "shared_shift_fallback"
        runtime = item["runtime"]
        runtime["quality_refiner"].update(quality_calls_completed=1, witness_audits_passed=1)
        runtime["causal_shift_fallback_traces"] = [
            dict(
                response_evaluations=4,
                fallback_rollouts=1,
                audit_status="passed",
                primary_initialization=dict(start_lp_attempts=1),
            )
        ]
        runtime["shared_shift_traces"] = [
            dict(
                shared_search=dict(
                    primary_responses=2,
                    fallback_responses=1,
                    fallback_lp_attempts=0,
                    response_budget=320,
                    feasibility_iteration_budget=60,
                    total_iteration_budget=120,
                ),
                primary_response_evaluations=3,
                normalized_status="two_step_no_witness",
                shifted_rollouts=1,
                response_evaluations=4,
                selected_scope="shifted_plan_first_pose",
                audit_status="passed",
            )
        ]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["shifted_rollouts"] == counts["first_selected"] == 2 and counts["start_lp_calls"] == 2
    branches[0]["runtime"]["shared_shift_traces"][0]["response_evaluations"] = 321
    with pytest.raises(ValueError, match="Shared budgets"):
        runner.summarize_phase(branches, "producer", config)
    monkeypatch.setitem(sys.modules, runner.CHECKER_FORBIDDEN[-1], ModuleType(runner.CHECKER_FORBIDDEN[-1]))
    with pytest.raises(ValueError, match="candidates"):
        runner.replay_action([])


def test_clean_checker_can_bind_provenance_without_importing_candidates():
    code = (
        "import sys; from scripts.simulation.runtime import run_task2_shared_shift_feedback as r; "
        "r.load_inputs(); r.source_refs(); "
        "assert all(name not in sys.modules for name in r.CHECKER_FORBIDDEN)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=runner.REPO,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
