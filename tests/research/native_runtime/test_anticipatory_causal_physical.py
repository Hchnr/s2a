"""The physical probe must bind only completed history and a fixed action budget."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime.run_anticipatory_causal_physical import bind_after_completed


class FakeAdapter:
    def __init__(self):
        self.events, self.calls = [], 0

    def observe_completed(self, step, predicted, actual, *, next_pose_index):
        self.events.append((step, predicted.copy(), actual.copy(), next_pose_index))

    def solve(self, *_args, **_kwargs):
        self.calls += 1
        return {"witness_found": True, "response_evaluations": 1}

    def audit(self, *_args, **_kwargs):
        return {"passed": True}


def period(step=120):
    return {
        "step": step,
        "decision": {"quality": {"verification": {"passed": True}, "first_tcp_error": np.zeros(6)}},
        "tcp_error": np.array([0.001, 0, 0, 0, 0, 0]),
        "task_success": object(),
    }


def test_binding_uses_completed_error_and_enforces_two_action_budget():
    baseline, adapter, receipt = object(), FakeAdapter(), {}
    refiner = SimpleNamespace(solver=baseline, auditor=None, max_quality_response_evaluations=160)
    bound = bind_after_completed(refiner, period(), receipt, adapter=adapter, expected_solver=baseline)
    assert bound is adapter and adapter.events[0][0] == 120 and adapter.events[0][3] == 1
    np.testing.assert_array_equal(adapter.events[0][2], [0.001, 0, 0, 0, 0, 0])
    assert refiner.max_quality_response_evaluations == 320
    assert refiner.solver()["witness_found"] is True
    assert refiner.solver()["witness_found"] is True
    stopped = refiner.solver()
    assert stopped["witness_found"] is False and stopped["status"] == "registered_two_period_limit"
    assert adapter.calls == 2
    assert receipt["causal_lookahead_binding"]["limit_stops"] == 1


def test_binding_rejects_future_or_unverified_history():
    baseline, adapter = object(), FakeAdapter()
    refiner = SimpleNamespace(solver=baseline, auditor=None, max_quality_response_evaluations=160)
    with pytest.raises(ValueError, match="actual completed step120"):
        bind_after_completed(refiner, period(121), {}, adapter=adapter, expected_solver=baseline)
    assert adapter.events == [] and refiner.solver is baseline


def test_checker_import_does_not_load_candidate_policy():
    script = (
        "from scripts.simulation.runtime.run_anticipatory_causal_physical import checker_isolation; checker_isolation()"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
