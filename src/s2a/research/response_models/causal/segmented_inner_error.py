"""Expose segment-relative counts without discarding completed inner history."""

from s2a.research.response_models.causal.causal_inner_error import CausalInnerError


class SegmentedInnerHistory:
    """Own a single causal history and hand out one current segment view at a time.

    The caller must independently align the physical clock, tracking and gripper
    transactions before opening a new segment. This class only protects the inner
    error history and is not an action-dispatch or physical-safety gate.
    """

    def __init__(self, history=None):
        if history is None:
            history = CausalInnerError()
        if type(history) is not CausalInnerError:
            raise ValueError("One underlying, non-segmented inner history is required")
        self._history = history
        self._active = None

    def total_snapshot(self):
        """Return the global completed history, without rebasing its count."""
        return self._history.snapshot()

    def begin_segment(self, start_step):
        view = self.prepare_segment(start_step)
        self.activate_segment(view)
        return view

    def prepare_segment(self, start_step):
        """Create an inactive view without changing the currently usable view."""
        if type(start_step) is not int or start_step < 0:
            raise ValueError("A nonnegative absolute starting step is required")
        history = self._history
        if history._prediction is not None or history._period._pending is not None:
            raise ValueError("The preceding prediction must physically complete before a new segment")
        past = history.snapshot()
        if past["last_completed_step"] is not None and start_step != past["last_completed_step"] + 1:
            raise ValueError("Segments must continue at the next absolute step")
        return _SegmentView(self, start_step, past["completed_periods"], past["last_completed_step"], self._active)

    def activate_segment(self, view):
        """Switch only if no history changed after the view was prepared."""
        if type(view) is not _SegmentView or view._owner is not self or view._predecessor is not self._active:
            raise ValueError("One newly prepared view of this history is required")
        history = self._history
        past = history.snapshot()
        if (
            history._prediction is not None
            or history._period._pending is not None
            or past["completed_periods"] != view._initial_count
            or past["last_completed_step"] != view._prior_last
        ):
            raise ValueError("History changed while preparing the new segment")
        self._active = view


class _SegmentView(CausalInnerError):
    """Adapt a global history to the existing refiner's local offset contract."""

    def __init__(self, owner, start_step, initial_count, prior_last, predecessor):
        self._owner, self._start_step, self._initial_count = owner, start_step, initial_count
        self._prior_last, self._predecessor = prior_last, predecessor

    def _current(self):
        if self._owner._active is not self:
            raise ValueError("This inner-history segment has expired")
        history = self._owner._history
        past = history.snapshot()
        count = past["completed_periods"] - self._initial_count
        if count < 0 or (count and past["last_completed_step"] != self._start_step + count - 1):
            raise ValueError("Underlying history no longer matches this segment")
        return history, past, count

    @property
    def _prediction(self):
        history, _past, _count = self._current()
        return history._prediction

    def snapshot(self):
        _history, past, count = self._current()
        past["total_completed_periods"] = past["completed_periods"]
        past["completed_periods"] = count
        return past

    def stage(self, step, predicted_inner):
        history, _past, count = self._current()
        if type(step) is not int or step != self._start_step + count:
            raise ValueError("Prediction must match the next absolute segment step")
        history.stage(step, predicted_inner)

    def complete(self, step, actual_inner):
        history, _past, count = self._current()
        if type(step) is not int or step != self._start_step + count:
            raise ValueError("Observation must match the staged absolute segment step")
        return history.complete(step, actual_inner)
