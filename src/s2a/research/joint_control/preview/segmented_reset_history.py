"""Adapt an empty reset history to causal reference segments."""

from dataclasses import dataclass

import numpy as np

from s2a.research.joint_control.preview.reset_history import make_reset_preview_history
from s2a.research.joint_control.preview.tracking_preview_transaction import TrackingPreviewTransaction
from s2a.research.response_models.causal.causal_inner_error import CausalInnerError
from s2a.research.response_models.causal.coupled_response_history import CoupledResponseHistory
from s2a.research.response_models.causal.segmented_inner_error import SegmentedInnerHistory


@dataclass(frozen=True)
class SegmentedResetHistory:
    """Retain one owner and the current view without inventing prior observations."""

    previous_command: np.ndarray
    coupled: CoupledResponseHistory
    tracking: TrackingPreviewTransaction
    inner_error: CausalInnerError
    owner: SegmentedInnerHistory


def segment_empty_reset_history(history):
    """Bind an existing, still-empty reset history to its first segment."""
    if type(history) is not CausalInnerError:
        raise ValueError("One unsegmented reset history is required")
    current = history.snapshot()
    if (
        current["available"] is not False
        or current["completed_periods"] != 0
        or current["last_completed_step"] is not None
        or history._prediction is not None
        or np.any(current["inner_position_error_rad"] != 0)
    ):
        raise ValueError("Only an empty reset history may start the first segment")
    owner = SegmentedInnerHistory(history)
    return owner, owner.begin_segment(0)


def make_segmented_reset_history(measured_site, measured_twist, *, refiner_factory):
    """Reuse reset initialization while binding its refiner to the first view.

    History switching and actual physical admission remain the caller's duty.
    No completed action, plan, inner profile, or observation is invented here.
    """
    if not callable(refiner_factory):
        raise TypeError("A factory for this episode's causal refiner is required")
    owner = view = None

    def bind_view(history):
        nonlocal owner, view
        owner, view = segment_empty_reset_history(history)
        return refiner_factory(view)

    reset = make_reset_preview_history(measured_site, measured_twist, refiner_factory=bind_view)
    return SegmentedResetHistory(reset.previous_command, reset.coupled, reset.tracking, view, owner)
