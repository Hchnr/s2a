"""Create isolated causal preview histories at a registered controller reset."""

from dataclasses import dataclass

import numpy as np

from s2a.research.joint_control.preview.tracking_preview_transaction import TrackingPreviewTransaction
from s2a.research.response_models.causal.causal_inner_error import CausalInnerError
from s2a.research.response_models.causal.coupled_response_history import CoupledResponseHistory
from s2a.research.response_models.causal.runtime_response_history import RuntimeResponseHistory


@dataclass(frozen=True)
class ResetPreviewHistory:
    """Separate reset values; zero innovation is an unknown-history initialization."""

    previous_command: np.ndarray
    coupled: CoupledResponseHistory
    tracking: TrackingPreviewTransaction
    inner_error: CausalInnerError


def make_reset_preview_history(measured_site, measured_twist, *, refiner_factory):
    """Bind fresh histories to one reset endpoint without inventing a prior plan.

    The zero previous command is the registered native-controller reset convention;
    the caller must separately verify the actual reset and preserve gripper motor state.
    """
    site, twist = RuntimeResponseHistory.endpoint(measured_site, measured_twist)
    if not callable(refiner_factory):
        raise TypeError("A factory for this episode's causal refiner is required")
    inner_error = CausalInnerError()
    refiner = refiner_factory(inner_error)
    if not callable(refiner):
        raise TypeError("The refiner factory must return a callable")
    coupled = CoupledResponseHistory()
    tracking = TrackingPreviewTransaction(refiner=refiner)
    for history in (coupled, tracking.history):
        history.expected_pose = site.copy()
        history.expected_twist = twist.copy()
    previous = np.zeros(7)
    previous.setflags(write=False)
    return ResetPreviewHistory(previous, coupled, tracking, inner_error)
