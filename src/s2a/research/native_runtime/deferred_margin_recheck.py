"""Bounded diagnostic for one previously audited margin refresh."""

import copy

from s2a.research.native_runtime.reference_choice_comparison import quality_stop


class DeferredMarginRecheck:
    """Defer at most one specific margin trigger; preserve all other stops.

    This is an experimental intervention, not a replacement for the original
    refresh rule or an authorization to dispatch an action. The caller must
    still obtain and audit a new action from the actual current state.
    """

    def __init__(self, *, target_step):
        if type(target_step) is not int or target_step < 1:
            raise ValueError("A positive fixed control step is required")
        self.target_step = target_step
        self._events = []

    @property
    def events(self):
        return copy.deepcopy(self._events)

    def __call__(self, previous_period, current_step, trigger):
        stop = quality_stop(previous_period, current_step, trigger)
        if (
            current_step == self.target_step
            and not self._events
            and stop is not None
            and stop["stop"]["trigger"]["reason"] == "quality_margin_consumed"
        ):
            self._events.append(dict(current_step=current_step, original_stop=copy.deepcopy(stop)))
            return None
        return stop
