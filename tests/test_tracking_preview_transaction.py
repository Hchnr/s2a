"""Check command identity and causal completion at the native-loop boundary."""

import numpy as np
import pytest

from s2a.tracking_preview_transaction import TrackingPreviewTransaction


def staged_refiner(*args, **kwargs):
    history = args[-1]
    plan = np.array([[0.1, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]])
    history.stage_plan(plan)
    return dict(command=plan[0].copy(), planned_commands=plan, nominal_response=np.zeros(12))


def prepare(transaction):
    return transaction.prepare(
        None,
        {},
        np.eye(4),
        np.zeros(6),
        np.eye(4)[None],
        np.eye(4),
        0,
        np.zeros((1, 6)),
        np.zeros(6),
        {},
        site_from_tcp=np.eye(4),
    )


def test_only_actual_completed_period_updates_history():
    tx = TrackingPreviewTransaction(refiner=staged_refiner)
    result = prepare(tx)
    assert tx.history.shifted(2) is None
    with pytest.raises(ValueError, match="committed"):
        tx.observe(np.eye(4), np.zeros(6))
    with pytest.raises(ValueError, match="differs"):
        tx.commit(np.zeros(6))
    tx.commit(result["command"])
    with pytest.raises(ValueError, match="Observe"):
        prepare(tx)
    measured = np.eye(4)
    measured[0, 3] = 0.001
    twist = np.array([0.2, 0, 0, 0, 0, 0])
    innovation = tx.observe(measured, twist)
    np.testing.assert_allclose(innovation, np.r_[0.001, np.zeros(5), 0.01, np.zeros(5)], atol=1e-15, rtol=0)
    np.testing.assert_array_equal(tx.history.shifted(2), np.tile([0.2, 0, 0, 0, 0, 0], (2, 1)))
    assert tx.history.completed_periods == 1
    tx.reset()
    assert tx.history.completed_periods == 0 and tx.history.shifted(2) is None
    np.testing.assert_array_equal(tx.history.innovation, np.zeros(12))


def test_returned_log_cannot_change_committed_prediction():
    tx = TrackingPreviewTransaction(refiner=staged_refiner)
    result = prepare(tx)
    result["command"][0] = 0.9
    result["nominal_response"][:] = 9
    with pytest.raises(ValueError, match="differs"):
        tx.commit(result["command"])
    tx.commit([0.1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(tx.observe(np.eye(4), np.zeros(6)), np.zeros(12))


def test_failed_prepare_requires_explicit_case_reset():
    def bad_refiner(*args, **kwargs):
        args[-1].stage_plan(np.zeros((1, 6)))
        raise ValueError("failed optimization")

    tx = TrackingPreviewTransaction(refiner=bad_refiner)
    with pytest.raises(ValueError, match="failed optimization"):
        prepare(tx)
    assert tx.phase == "faulted"
    with pytest.raises(ValueError, match="reset"):
        prepare(tx)
    tx.reset()
    assert tx.phase == "ready" and tx.history.pending_plan is None
