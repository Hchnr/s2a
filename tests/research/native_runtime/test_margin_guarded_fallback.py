"""Matched static comparison binds prior inputs and audits selection scope."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_two_step_viability as original
from scripts.tracking.trajectory_search import compare_margin_guarded_fallback as runner


def test_four_prior_margin_and_guarded_decisions_are_bound():
    inputs = runner.load_inputs()
    assert [case["decision_spec"]["case_index"] for case in inputs["cases"]] == [2, 6, 10, 11]
    assert [case["margin_static"]["model_prefix_witness"] for case in inputs["cases"]] == [False, False, False, True]
    assert [case["guarded_static"]["model_prefix_witness"] for case in inputs["cases"]] == [False, False, True, True]
    assert runner.input_receipt(inputs)["prior_ref"] == runner.PRIOR


def test_isolated_analyzer_and_worker_keep_original_globals():
    analyzer, worker = runner.isolated_analyze(), runner.isolated_worker()
    assert analyzer.__code__ is original.analyze.__code__ and analyzer.__globals__ is not original.analyze.__globals__
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert analyzer.__globals__["solve_two_step_viability"] is runner.policy.solve_margin_guarded_fallback
    assert worker.__globals__["analyze"] is runner.analyze
    assert original.analyze.__globals__["solve_two_step_viability"] is original.solve_two_step_viability


def test_changed_input_fingerprint_is_rejected(monkeypatch):
    case = dict(margin_static=dict(input_sha256="a" * 64))
    monkeypatch.setattr(runner, "isolated_analyze", lambda: lambda *_args: {"input_sha256": "b" * 64})
    with pytest.raises(ValueError, match="byte-identical"):
        runner.analyze(case, {}, {})


@pytest.mark.parametrize("scope,expected", [("margin_two_step", False), ("guarded_two_step", True)])
def test_only_margin_scope_requires_independently_reconstructed_extra_margin(monkeypatch, scope, expected):
    errors = np.zeros((8, 6))
    errors[1, 0] = 0.0097
    prior = dict(input_sha256="a" * 64, seed_model_first_error=np.zeros(6), seed_observed_first_error=np.zeros(6))
    fake = dict(
        input_sha256="a" * 64,
        seed_model_first_error=np.zeros(6),
        seed_observed_first_error=np.zeros(6),
        solver=dict(best_joint_safe=dict(selection_scope=scope)),
        independent_audit=dict(status="passed", verification=dict(tcp_errors=errors)),
        model_prefix_witness=True,
    )
    monkeypatch.setattr(runner, "isolated_analyze", lambda: lambda *_args: fake.copy())
    result = runner.analyze(dict(margin_static=prior, guarded_static={}), {}, {})
    assert result["model_prefix_witness"] is expected
    assert result["independent_scope_audit"]["status"] == ("passed" if expected else "rejected")
