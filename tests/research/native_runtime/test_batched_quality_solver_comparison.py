"""Solver comparison must retain mismatches instead of claiming equivalence."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import compare_batched_quality_solver as comparison


def witness():
    return {
        "status": "model_witness",
        "witness_found": True,
        "response_evaluations": 12,
        "planned_commands": np.zeros((8, 6)),
        "first_tcp_error": np.zeros(6),
        "objective": 0.25,
        "optimizer_success": True,
        "iterations": 3,
        "independent_audit": {"passed": True},
    }


def test_reports_equivalent_and_different_witnesses_without_retrying():
    first, second = witness(), witness()
    assert comparison.compare_runs(first, second, atol=1e-10)["equivalent_at_tolerance"]
    second["planned_commands"][0, 0] = 0.001
    result = comparison.compare_runs(first, second, atol=1e-10)
    assert not result["equivalent_at_tolerance"] and result["planned_command_max_abs_difference"] == 0.001
    second = copy.deepcopy(first)
    second.update(status="nonlinear_search_no_witness", witness_found=False)
    result = comparison.compare_runs(first, second, atol=1e-10)
    assert not result["same_status_and_witness"] and result["planned_command_max_abs_difference"] is None
    second = copy.deepcopy(first)
    second["independent_audit"]["passed"] = False
    assert not comparison.compare_runs(first, second, atol=1e-10)["equivalent_at_tolerance"]


def test_audit_rejection_is_recorded_and_unexpected_errors_remain_errors():
    ledger = dict.fromkeys(
        (
            "independent_audits_attempted",
            "independent_audits_completed",
            "independent_audits_passed",
            "independent_audits_failed",
        ),
        0,
    )

    def reject():
        raise ValueError("outside the audited constraint")

    result = comparison.record_audit(reject, ledger)
    assert result["passed"] is False and result["error"]["type"] == "ValueError"
    assert ledger["independent_audits_attempted"] == ledger["independent_audits_completed"] == 1
    assert ledger["independent_audits_failed"] == 1 and ledger["independent_audits_passed"] == 0

    def crash():
        raise RuntimeError("unexpected failure")

    with pytest.raises(RuntimeError):
        comparison.record_audit(crash, ledger)
    assert ledger["independent_audits_attempted"] == 2 and ledger["independent_audits_completed"] == 1
