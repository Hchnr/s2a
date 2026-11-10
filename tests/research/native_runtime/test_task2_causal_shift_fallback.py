"""Feedback binding preserves the prior source and replay boundaries."""

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.simulation.runtime import run_task2_causal_shift_fallback as runner
from scripts.simulation.runtime import run_task2_earlier_preview_feedback as original
from tests.research.native_runtime.test_task2_earlier_restoration_feedback import branch


def test_isolated_worker_uses_fallback_factory_without_modifying_original():
    worker = runner.isolated_worker()
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert worker.__globals__["ROOT"] == runner.ROOT
    assert worker.__globals__["full"].make_full_refiner is runner.make_fallback_refiner


def test_replayed_inputs_bind_prior_guarded_feedback_and_static_qualification():
    inputs = runner.load_inputs()
    assert [item["spec"]["prefix_steps"] for item in inputs["cases"]] == [32, 80]
    receipt = runner.input_receipt(inputs)
    assert receipt["static_ref"] == runner.STATIC and receipt["prior_producer_ref"] == runner.PRIOR_PRODUCER


def test_fallback_factory_attaches_the_actual_transaction_history(monkeypatch):
    from s2a.research.trajectory_search.adaptive_plan_history import AdaptivePlanHistory

    history = AdaptivePlanHistory()
    policy = SimpleNamespace(tracking=SimpleNamespace(history=history))
    captured = {}

    def build(_env, _case, receipt, factory):
        captured["adapter"] = factory(receipt)
        return policy, "inner_history"

    monkeypatch.setattr(runner.builder, "build_refiner", build)
    assert runner.make_fallback_refiner(None, None, {}) == (policy, "inner_history")
    assert captured["adapter"]._history is history


def test_fallback_costs_include_the_extra_rollout_and_enforce_budget(monkeypatch):
    config = runner.read_config()[1]
    branches = [branch(10, 1), branch(11, 1)]
    for item in branches:
        item["branch"] = "causal_shift_fallback"
        item["runtime"]["quality_refiner"].update(quality_calls_completed=1, witness_audits_passed=1)
        item["runtime"]["causal_shift_fallback_traces"] = [
            dict(
                primary_response_evaluations=3,
                fallback_rollouts=1,
                response_evaluations=4,
                primary_initialization=dict(start_lp_attempts=1),
                selected_scope="first_only",
                primary_status="two_step_no_witness",
                audit_status="passed",
            )
        ]
    counts = runner.summarize_phase(branches, "producer", config)
    assert counts["fallback_rollouts"] == counts["fallback_selected"] == 2 and counts["start_lp_calls"] == 2
    branches[0]["runtime"]["causal_shift_fallback_traces"][0]["response_evaluations"] = 321
    with pytest.raises(ValueError, match="accounting"):
        runner.summarize_phase(branches, "producer", config)
    monkeypatch.setitem(sys.modules, runner.CHECKER_FORBIDDEN[-1], ModuleType(runner.CHECKER_FORBIDDEN[-1]))
    with pytest.raises(ValueError, match="candidate policies"):
        runner.replay_action([])


def test_checker_can_bind_provenance_without_importing_candidate_modules():
    code = (
        "import sys; from scripts.simulation.runtime import run_task2_causal_shift_fallback as r; "
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
