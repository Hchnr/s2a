"""The guarded comparison must retain the prior inputs and frozen worker."""

import pytest

from scripts.tracking.trajectory_search import audit_two_step_viability as previous
from scripts.tracking.trajectory_search import compare_guarded_two_step_search as runner


def test_isolated_analyzer_and_worker_do_not_modify_original_globals():
    analyzer, worker = runner.isolated_analyze(), runner.isolated_worker()
    assert analyzer.__code__ is previous.analyze.__code__ and analyzer.__globals__ is not previous.analyze.__globals__
    assert worker.__code__ is previous.worker.__code__ and worker.__globals__ is not previous.worker.__globals__
    assert analyzer.__globals__["solve_two_step_viability"] is runner.solve_guarded_two_step
    assert worker.__globals__["analyze"] is runner.analyze


def test_prior_four_decisions_and_recorded_outcomes_are_bound():
    cases = runner.load_inputs()["cases"]
    assert [case["decision_spec"]["case_index"] for case in cases] == [2, 6, 10, 11]
    assert [case["prior_static"]["model_prefix_witness"] for case in cases] == [False, False, False, True]
    assert all(len(case["prior_static"]["input_sha256"]) == 64 for case in cases)


def test_changed_solver_input_fingerprint_is_rejected(monkeypatch):
    case = dict(prior_static=dict(input_sha256="a" * 64))
    monkeypatch.setattr(runner, "isolated_analyze", lambda: lambda *_args: {"input_sha256": "b" * 64})
    with pytest.raises(ValueError, match="byte-identical"):
        runner.analyze(case, {}, {})
