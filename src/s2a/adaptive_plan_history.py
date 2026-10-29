"""Remember planned commands only after their actual first period is observed."""

import numpy as np

from s2a.geometry import finite
from s2a.joint_limit_native import JointLimitResponseHistory


class AdaptivePlanHistory(JointLimitResponseHistory):
    def reset(self):
        super().reset()
        self.last_plan = None
        self.pending_plan = None

    def stage_plan(self, plan):
        plan = finite(plan)
        if plan.ndim != 2 or plan.shape[1] != 6 or not 1 <= len(plan) <= 8 or self.pending_plan is not None:
            raise ValueError("One unstaged supported complete plan required")
        self.pending_plan = plan.copy()

    def observe_completed(self, pose, twist):
        result = super().observe_completed(pose, twist)
        if self.pending_plan is not None:
            self.last_plan = self.pending_plan.copy()
            self.pending_plan = None
        return result

    def shifted(self, horizon):
        if not 1 <= horizon <= 8:
            raise ValueError("Supported horizon required")
        if self.last_plan is None or len(self.last_plan) < 2:
            return None
        tail = self.last_plan[1:]
        return np.vstack((tail, np.tile(tail[-1], (max(0, horizon - len(tail)), 1))))[:horizon].copy()
