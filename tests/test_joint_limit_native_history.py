"""Executed-prediction history must stay separate and require completed endpoints."""

import numpy as np
import pytest

from s2a.joint_limit_native import JointLimitResponseHistory


def test_completed_history_uses_executed_prediction_and_rejects_pending_replan():
    history = JointLimitResponseHistory()
    pose, twist, prediction = np.eye(4), np.zeros(6), np.zeros(12)
    prediction[0] = 0.01
    history.commit_prediction(pose, twist, prediction)
    with pytest.raises(ValueError, match="observed before replanning"):
        history.current(pose, twist)
    after = pose.copy()
    after[0, 3] = 0.012
    innovation = history.observe_completed(after, twist)
    np.testing.assert_allclose(innovation, [0.002] + [0] * 11, atol=1e-12, rtol=0)
    other = JointLimitResponseHistory()
    np.testing.assert_array_equal(other.current(pose, twist), np.zeros(12))
