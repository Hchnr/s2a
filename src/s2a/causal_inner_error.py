"""Past-only empirical position errors indexed by inner-period time."""

import numpy as np

from s2a.causal_joint_error import CausalJointError
from s2a.geometry import finite


class CausalInnerError:
    """Keep a nondecreasing 25-sample profile, without a future guarantee."""

    def __init__(self, joints=7):
        self._period = CausalJointError(joints)
        self._profile = np.zeros((25, joints))
        self._prediction = None

    def snapshot(self):
        result = self._period.snapshot()
        result["inner_position_error_rad"] = np.maximum.accumulate(self._profile, axis=0)
        return result

    def stage(self, step, predicted_inner):
        predicted = finite(predicted_inner, self._profile.shape).copy()
        self._period.stage(step, predicted)
        self._prediction = predicted

    def complete(self, step, actual_inner):
        actual = finite(actual_inner, self._profile.shape)
        residual = self._period.complete(step, actual)
        self._profile = np.maximum(self._profile, abs(actual - self._prediction))
        self._prediction = None
        return residual
