"""Deferral stages and commits only the action that is physically executed."""

import copy

import numpy as np
import pytest

from s2a.research.gripper.defer_close_transaction import DeferringLoadedGripperTransaction
from s2a.research.gripper.loaded_gripper_transaction import LoadedGripperTransaction
from tests.research.gripper.test_defer_close_release import sample
from tests.research.gripper.test_loaded_gripper_transaction import fixture


def trigger_fixture():
    original, _ = fixture()
    controller = copy.deepcopy(original._controller)
    controller.latch._closing = True
    controller.latch._reference_min = 0.1
    poses, openings, index, tcp, measured, motor, _baseline, _closing = sample()
    return controller, (measured, motor, 0.0, poses, openings, index, tcp)


def test_deferral_commits_only_after_actual_hold_and_preserves_latch_history():
    controller, args = trigger_fixture()
    transaction = DeferringLoadedGripperTransaction(controller)
    before = transaction.state()
    decision = transaction.prepare(*args, previous_command=0.0, dt=0.05)
    assert decision["detail"]["baseline"]["command"] == 1.0
    assert decision["detail"]["deferral"]["deferred"] is True
    assert decision["command"] == decision["detail"]["command"] == 0.0
    assert transaction.state()["latch"] == before["latch"] and transaction.state()["staged"]
    with pytest.raises(ValueError, match="differs"):
        transaction.commit(decision, actual_command=1.0, completed_steps=1)
    transaction.commit(decision, actual_command=0.0, completed_steps=1)
    assert transaction.state()["latch"]["closing"] is True
    assert transaction.state()["staged"] is False


def test_nontriggering_path_matches_original_transaction_and_abort():
    controller, args = trigger_fixture()
    changed = list(args)
    changed[1] = np.array([0.0, 0.0])
    base = LoadedGripperTransaction(controller)
    candidate = DeferringLoadedGripperTransaction(controller)
    original = base.prepare(*changed, previous_command=0.0, dt=0.05)
    selected = candidate.prepare(*changed, previous_command=0.0, dt=0.05)
    assert selected["detail"]["deferral"]["deferred"] is False
    assert selected["command"] == original["command"]
    assert selected["detail"]["baseline"] == original["detail"]
    candidate.abort(selected)
    assert not candidate.state()["staged"]
    selected = candidate.prepare(*changed, previous_command=0.0, dt=0.05)
    candidate.commit(selected, actual_command=selected["command"], completed_steps=1)
    base.commit(original, actual_command=original["command"], completed_steps=1)
    assert candidate.state() == base.state()


def test_partial_execution_faults_the_transaction():
    controller, args = trigger_fixture()
    transaction = DeferringLoadedGripperTransaction(controller)
    decision = transaction.prepare(*args, previous_command=0.0, dt=0.05)
    transaction.abort(decision, partial_execution=True)
    with pytest.raises(ValueError, match="Reconstruct history"):
        transaction.prepare(*args, previous_command=0.0, dt=0.05)
