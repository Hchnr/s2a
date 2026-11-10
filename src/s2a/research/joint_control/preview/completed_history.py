"""Restore preview histories solely from a completed control period."""

from dataclasses import dataclass

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.limits.joint_limit_native import JointLimitResponseHistory
from s2a.research.joint_control.preview.tracking_preview_transaction import TrackingPreviewTransaction
from s2a.research.response_models.causal.coupled_response_history import CoupledResponseHistory
from s2a.research.response_models.causal.runtime_response_history import RuntimeResponseHistory


@dataclass(frozen=True)
class CompletedHistory:
    """Minimal causal values committed and observed before the next decision."""

    previous_command: np.ndarray
    coupled_innovation: np.ndarray
    tracking_innovation: np.ndarray
    last_plan: np.ndarray
    completed_periods: int

    def __post_init__(self):
        previous = finite(self.previous_command, (7,)).copy()
        coupled = finite(self.coupled_innovation, (12,)).copy()
        tracking = finite(self.tracking_innovation, (12,)).copy()
        plan = finite(self.last_plan).copy()
        if type(self.completed_periods) is not int or not 0 < self.completed_periods <= 600:
            raise ValueError("A positive completed-period count is required")
        if plan.ndim != 2 or plan.shape[1] != 6 or not 1 <= len(plan) <= 8:
            raise ValueError("One completed bounded arm plan is required")
        if np.max(np.abs(previous)) > 1 + 1e-10 or np.max(np.abs(plan)) > 1 + 1e-10:
            raise ValueError("Completed command and plan must fit the controller box")
        if not np.array_equal(plan[0], previous[:6]):
            raise ValueError("Completed plan must begin with the actual previous command")
        if len(plan) > 1 and np.max(np.abs(np.diff(plan, axis=0))) > 0.25 + 1e-10:
            raise ValueError("Completed plan exceeds the original action rate")
        for name, value in (
            ("previous_command", previous),
            ("coupled_innovation", coupled),
            ("tracking_innovation", tracking),
            ("last_plan", plan),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def recompute_completed_innovation(previous_site, previous_twist, prediction, current_site, current_twist):
    """Recover one completed innovation from its prediction and measured endpoint."""
    history = JointLimitResponseHistory()
    history.commit_prediction(previous_site, previous_twist, prediction)
    return history.observe_completed(current_site, current_twist)


def restore_completed_history(completed, current_site, current_twist):
    """Return two independent ready histories at the just-observed endpoint."""
    if not isinstance(completed, CompletedHistory):
        raise TypeError("Explicit completed history is required")
    site, twist = RuntimeResponseHistory.endpoint(current_site, current_twist)
    coupled = CoupledResponseHistory()
    coupled.innovation = completed.coupled_innovation.copy()
    coupled.expected_pose, coupled.expected_twist = site.copy(), twist.copy()
    coupled.completed_periods = completed.completed_periods
    tracking = TrackingPreviewTransaction()
    tracking.history.innovation = completed.tracking_innovation.copy()
    tracking.history.expected_pose, tracking.history.expected_twist = site.copy(), twist.copy()
    tracking.history.completed_periods = completed.completed_periods
    tracking.history.last_plan = completed.last_plan.copy()
    return coupled, tracking
