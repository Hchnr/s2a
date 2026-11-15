"""Causal preview history follows only audited and completed physical periods."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.causal_lookahead_feedback import CausalLookaheadSolverAudit
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def started_adapter():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.02
    adapter = CausalLookaheadSolverAudit(position_cap_m=0.009, rotation_cap_rad=0.045)
    adapter.observe_completed(120, np.zeros(6), np.array([0.001, -0.001, 0, 0, 0, 0]), next_pose_index=1)
    return adapter, args, audit_args


def test_next_solve_requires_the_audited_action_to_finish():
    adapter, args, audit_args = started_adapter()
    witness = adapter.solve(*args)
    assert witness["witness_found"] is True
    with pytest.raises(ValueError, match="previous quality witness"):
        adapter.solve(*args)
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked["passed"] and checked["causal_corrected_pose_index"] == 1
    with pytest.raises(ValueError, match="fully observed causal history"):
        adapter.solve(*args)
    predicted = checked["tcp_errors"][0]
    with pytest.raises(ValueError, match="matching previously audited action"):
        adapter.observe_completed(122, predicted, predicted, next_pose_index=0)
    adapter.observe_completed(121, predicted, predicted + [0.0002, 0, 0, 0, 0, 0], next_pose_index=0)
    assert adapter.history.completed == 2
    second = adapter.solve(*args)
    assert second["witness_found"] is True
    checked_second = adapter.audit(second, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert checked_second["causal_corrected_pose_index"] == 0


def test_observed_pose_violation_stops_before_the_next_candidate():
    adapter, args, audit_args = started_adapter()
    witness = adapter.solve(*args)
    checked = adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    actual = np.array([0.02, 0, 0, 0, 0, 0])
    adapter.observe_completed(121, checked["tcp_errors"][0], actual, next_pose_index=0)
    stopped = adapter.solve(*args)
    assert stopped["witness_found"] is False
    assert stopped["status"] == "observed_pose_limit_violation"
    assert stopped["response_evaluations"] == 0


def test_tampered_causal_witness_cannot_stage_a_future_observation():
    adapter, args, audit_args = started_adapter()
    witness = adapter.solve(*args)
    witness["innovation_corrected_tcp_error"][0] += 0.1
    with pytest.raises(AssertionError):
        adapter.audit(witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)))
    assert adapter.history.completed == 1
    with pytest.raises(ValueError, match="matching previously audited action"):
        adapter.observe_completed(121, np.zeros(6), np.zeros(6), next_pose_index=0)
