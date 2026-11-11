"""Cross-task feedback preserves causal history, budgets and independent replay."""

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime import run_replan_shared_shift_feedback as runner
from tests.research.native_runtime.test_task2_earlier_restoration_feedback import branch


def test_factory_attaches_branch_history_and_passes_only_policy_inputs(monkeypatch):
    from s2a.research.trajectory_search.adaptive_plan_history import AdaptivePlanHistory

    history = AdaptivePlanHistory()
    policy = SimpleNamespace(tracking=SimpleNamespace(history=history))
    case, sanitized, captured = object(), {"generated_reference": True}, {}
    monkeypatch.setattr(runner.binding, "policy_case", lambda value: sanitized if value is case else None)

    def build(_env, selected, receipt, factory):
        captured.update(selected=selected, adapter=factory(receipt))
        return policy, "inner_history"

    monkeypatch.setattr(runner.builder, "build_refiner", build)
    assert runner.make_shared_refiner(None, case, {}) == (policy, "inner_history")
    assert captured["selected"] is sanitized and captured["adapter"]._history is history


def test_actual_inner_loss_blocks_dispatch_and_scores_only_after_completion(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "make_shared_refiner", lambda *_: (object(), object()))

    def callbacks(_policy, _history, _receipt, _start):
        def choose(_q, _limits, _jacobian, _previous, offset):
            calls.append(("select", offset))
            return {"command": np.zeros(7), "newly_selected": True}

        def done(_env, _command, offset):
            calls.append(("observe", offset))
            return {"completed": offset}

        return choose, done

    def score_after(done, _receipt, _case):
        return lambda env, command, offset: {"policy_history": done(env, command, offset)}

    monkeypatch.setattr(runner.quality, "quality_callbacks", callbacks)
    monkeypatch.setattr(runner.previous, "after_quality_completed", score_after)
    q, limits, jac, previous = np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7)), np.zeros(7)
    receipt = {"final_prefix_inner": {"below_buffer_samples": 1}, "continuation_partial": {"periods": []}}
    action, _ = runner.feedback_callbacks(None, {}, receipt)
    assert action(q, limits, jac, previous, 0)["stop"]["reason"] == "observed_joint_buffer_loss"
    assert calls == []

    receipt["final_prefix_inner"]["below_buffer_samples"] = 0
    action, completed = runner.feedback_callbacks(None, {}, receipt)
    assert "command" in action(q, limits, jac, previous, 0)
    samples = np.zeros((25, 7))
    samples[13, 2] = 0.96
    receipt["continuation_partial"]["periods"].append({"actual_inner": [{"q": row} for row in samples]})
    score = completed(None, previous, 0)
    assert score["policy_history"] == {"completed": 0} and score["completed_buffer_gate"]["below_buffer"]
    assert action(q, limits, jac, previous, 1)["stop"]["reason"] == "observed_joint_buffer_loss"
    assert calls == [("select", 0), ("observe", 0)]


def test_shared_costs_and_checker_isolation(monkeypatch):
    config = runner.binding.read_config()[1]
    branches = [branch(2, 1), branch(6, 1)]
    for item in branches:
        item["branch"] = "shared_shift_quality_new"
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


@pytest.mark.parametrize("phase", ["producer", "checker"])
def test_worker_binds_prefix_capture_selected_reference_and_replay(monkeypatch, phase):
    from s2a.research.native_runtime import development_prefix_runtime

    seen, published = {}, []
    case = {
        "spec": {"case_index": 2},
        "commands": [np.zeros(7)] * 16,
        "tool": np.eye(4),
        "config": {},
        "record": {},
        "record_xml": "xml",
        "saved": {},
        "receipts": {},
        "controller": {},
    }
    targets = np.tile(np.eye(4), (8, 1, 1))
    selected = {"fixed_targets": targets, "intervention_spec": {"target": "target"}}
    expected = [{"command": [0.0] * 7, "completed_feedback": {"reference_errors": {}}}]
    producer = {
        "complete": True,
        "passed": True,
        "runtime": {"continuation": {"periods": expected, "stop": None}, "final_prefix_inner": {"marker": 1}},
    }
    registration = {
        "inputs": {"bound": True},
        "source_refs": ["sources"],
        "producer_branch_refs": {runner.branch_key(2, runner.BRANCHES[0][1]): "producer"},
    }
    inputs = {"cases": [case], "asset_refs": [], "cv2_admission": {"cv2_probe_ref": 1, "cv2_runtime_source_refs": 2}}
    monkeypatch.setattr(runner.signal, "signal", lambda *_: None)
    monkeypatch.setattr(runner.prefix, "environment", lambda: None)
    monkeypatch.setattr(runner.prefix, "reference", lambda path: str(path))
    monkeypatch.setattr(runner.prefix, "read", lambda ref: producer if ref == "producer" else registration)
    monkeypatch.setattr(runner.binding, "load_inputs", lambda: inputs)
    monkeypatch.setattr(runner.binding, "input_receipt", lambda _: {"bound": True})
    monkeypatch.setattr(runner, "source_refs", lambda: ["sources"])
    monkeypatch.setattr(runner.switch, "selected_case", lambda value, mode: selected if value is case else None)
    monkeypatch.setattr(runner, "restore_cv2_environment", lambda *_: {})
    monkeypatch.setattr(runner, "CHECKER_FORBIDDEN", ())
    monkeypatch.setattr(runner, "publish", lambda _path, data: published.append(data) or {"published": True})

    def action(*_args):
        return {"command": np.zeros(7), "newly_selected": phase == "producer"}

    def completed(_env, _command, offset):
        seen["completed_offset"] = offset
        return {"score": True}

    monkeypatch.setattr(runner, "feedback_callbacks", lambda *_: (action, completed))
    monkeypatch.setattr(runner, "replay_action", lambda value: seen.update(replayed=value) or action)
    monkeypatch.setattr(runner.common, "check_prefix_inner", lambda actual, old: seen.update(prefix=(actual, old)))

    def replay_score(_done, _receipt, scoring_case, recorded):
        seen["scoring_case"], seen["scoring_expected"] = scoring_case, recorded
        return completed

    monkeypatch.setattr(runner.previous, "after_quality_completed", replay_score)
    monkeypatch.setattr(
        runner,
        "current_joint_site_state",
        lambda _env: (np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7))),
    )

    def run_continuation(_env, _terminal, _observers, _receipt, **kwargs):
        assert kwargs["start_step"] == 16 and kwargs["targets"] is targets
        assert kwargs["target_object"] == "target" and kwargs["expected_periods"] is (
            expected if phase == "checker" else None
        )
        assert kwargs["stop_on_outside"] is True
        decision = kwargs["choose_action"](np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7)), np.zeros(7), 0)
        assert kwargs["on_completed"](None, decision["command"], 0) == {"score": True}
        return {"periods": [{"step": 16}], "stop": None}

    monkeypatch.setattr(runner, "run_continuation", run_continuation)

    def run_prefix(*_args, **kwargs):
        assert kwargs["continuation_budget"] == 8 and kwargs["capture_final_prefix_inner"] is True
        receipt = {"final_prefix_inner": {"marker": 1, "below_buffer_samples": 0}, "newly_selected_commands": 0}
        outcome = kwargs["continuation"](None, None, None, receipt)
        return {**receipt, "continuation": outcome}

    monkeypatch.setattr(development_prefix_runtime, "run_prefix_runtime", run_prefix)
    runner.worker(phase, 2, runner.BRANCHES[0][1])
    assert published[0]["complete"] is published[0]["passed"] is True and seen["completed_offset"] == 0
    if phase == "checker":
        assert seen["replayed"] is seen["scoring_expected"] is expected and seen["scoring_case"] is case
        assert seen["prefix"][1] == {"marker": 1}


def test_clean_checker_binds_inputs_without_loading_candidate_modules():
    code = (
        "import sys; from scripts.simulation.runtime import run_replan_shared_shift_feedback as r; "
        "r.binding.load_inputs(); r.source_refs(); "
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
