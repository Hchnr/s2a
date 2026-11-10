"""A gripper proposal must not commit until its actual period completes."""

import copy

import numpy as np
import pytest

from s2a.research.gripper.causal_loaded_history import reconstruct_loaded_history
from s2a.research.gripper.loaded_gripper_transaction import LoadedGripperTransaction
from tests.research.gripper.test_causal_loaded_history import historical_prefix


def fixture():
    rows, chunks, tool, parameters, original = historical_prefix(3)
    controller, _ = reconstruct_loaded_history(
        rows,
        chunks,
        tool,
        parameters,
        velocity_envelope=original["gripper_velocity_envelope"],
        gripper_config=original["gripper_config"],
    )
    row, chunk = rows[-1], chunks[0]
    args = (
        row["opening_before"],
        row["motor_target_before"],
        row["measured_period_velocity"],
        chunk["effective_targets_tcp"],
        chunk["openings"],
        row["chunk_physical_step"],
        np.asarray(row["measured_source"]) @ np.asarray(tool),
    )
    return LoadedGripperTransaction(controller), args


def test_prepare_does_not_change_committed_state_and_abort_is_retryable():
    transaction, args = fixture()
    before = transaction.state()
    decision = transaction.prepare(*args, previous_command=0.0, dt=0.05)
    assert transaction.state()["latch"] == before["latch"] and transaction.state()["staged"]
    with pytest.raises(ValueError, match="Commit or abort"):
        transaction.prepare(*args, previous_command=0.0, dt=0.05)
    transaction.abort(decision)
    assert transaction.state() == before
    repeated = transaction.prepare(*args, previous_command=0.0, dt=0.05)
    assert repeated == decision
    transaction.commit(repeated, actual_command=repeated["command"], completed_steps=1)
    assert not transaction.state()["staged"]


def test_only_the_matching_unchanged_completed_decision_can_commit():
    transaction, args = fixture()
    decision = transaction.prepare(*args, previous_command=0.0, dt=0.05)
    with pytest.raises(ValueError, match="current staged"):
        transaction.commit(copy.deepcopy(decision), actual_command=decision["command"], completed_steps=1)
    with pytest.raises(ValueError, match="full completed"):
        transaction.commit(decision, actual_command=decision["command"], completed_steps=0)
    with pytest.raises(ValueError, match="differs"):
        transaction.commit(decision, actual_command=0.123, completed_steps=1)
    decision["detail"]["route"] = "fabricated"
    with pytest.raises(ValueError, match="differs"):
        transaction.commit(decision, actual_command=decision["command"], completed_steps=1)
    transaction.abort(decision)


def test_partial_execution_faults_the_episode_until_history_is_reconstructed():
    transaction, args = fixture()
    decision = transaction.prepare(*args, previous_command=0.0, dt=0.05)
    transaction.abort(decision, partial_execution=True)
    assert transaction.state()["faulted"]
    with pytest.raises(ValueError, match="Reconstruct history"):
        transaction.prepare(*args, previous_command=0.0, dt=0.05)
