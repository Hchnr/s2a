"""The margin comparison binds prior inputs and audits its own stricter gate."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_two_step_viability as original
from scripts.tracking.trajectory_search import compare_margin_two_step_search as runner


def test_guarded_decisions_and_recorded_results_are_bound():
    inputs = runner.load_inputs()
    assert [case["decision_spec"]["case_index"] for case in inputs["cases"]] == [2, 6, 10, 11]
    assert [case["guarded_static"]["model_prefix_witness"] for case in inputs["cases"]] == [False, False, True, True]
    assert all(len(case["guarded_static"]["input_sha256"]) == 64 for case in inputs["cases"])
    assert inputs["config"]["model_position_margin_m"] == 0.0005
    assert inputs["config"]["model_rotation_margin_rad"] == 0.003


def test_new_analyzer_and_worker_keep_frozen_global_namespaces():
    analyzer, worker = runner.isolated_analyze(), runner.isolated_worker()
    assert analyzer.__code__ is original.analyze.__code__ and analyzer.__globals__ is not original.analyze.__globals__
    assert worker.__code__ is original.worker.__code__ and worker.__globals__ is not original.worker.__globals__
    assert analyzer.__globals__["solve_two_step_viability"] is runner.margin.solve_margin_two_step
    assert worker.__globals__["analyze"] is runner.analyze
    assert original.analyze.__globals__["solve_two_step_viability"] is original.solve_two_step_viability


def test_config_rejects_changed_margin_before_input_loading(monkeypatch):
    real_read = runner.prefix.read

    def changed(ref, *args):
        result = real_read(ref, *args)
        result["model_position_margin_m"] = 0.0004
        return result

    monkeypatch.setattr(runner.prefix, "read", changed)
    with pytest.raises(ValueError, match="fixed model margins"):
        runner.read_config()


def test_changed_input_fingerprint_is_rejected(monkeypatch):
    case = dict(guarded_static=dict(input_sha256="a" * 64))
    monkeypatch.setattr(runner, "isolated_analyze", lambda: lambda *_args: {"input_sha256": "b" * 64})
    with pytest.raises(ValueError, match="byte-identical"):
        runner.analyze(case, {}, {})


@pytest.mark.parametrize("position,expected", [(0.0094, True), (0.0097, False)])
def test_independent_reconstruction_must_also_pass_stricter_margin(monkeypatch, position, expected):
    errors = np.zeros((8, 6))
    errors[1, 0] = position
    prior = dict(input_sha256="a" * 64, seed_model_first_error=np.zeros(6), seed_observed_first_error=np.zeros(6))
    fake = dict(
        input_sha256="a" * 64,
        seed_model_first_error=np.zeros(6),
        seed_observed_first_error=np.zeros(6),
        independent_audit=dict(status="passed", verification=dict(tcp_errors=errors)),
        model_prefix_witness=True,
    )
    monkeypatch.setattr(runner, "isolated_analyze", lambda: lambda *_args: fake.copy())
    result = runner.analyze(dict(guarded_static=prior), {}, {})
    assert result["external_two_step_witness"] is True
    assert result["model_prefix_witness"] is expected
    assert result["independent_margin_audit"]["status"] == ("passed" if expected else "rejected")
