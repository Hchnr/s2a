"""Bound current-image requests separated by variable, completed segments.

Physical receipts and trigger decisions are assertions by the owning runtime.
This protocol cannot establish actual dispatch, safety, or task success.
"""

import copy
import math

from s2a.research.wam.tracking.runtime.causal_stream_protocol import (
    SNAPSHOT_KEYS,
    CausalModelAdmission,
    CausalStreamLedger,
    _snapshot,
    _validate_request,
    model_inputs,
)


def _period_budget(max_periods):
    if type(max_periods) is not int or not 1 <= max_periods <= 600:
        raise ValueError("An explicit budget of 1–600 control periods is required")
    return max_periods


class AdaptiveStreamLedger(CausalStreamLedger):
    """Add action transactions and early refresh to the fixed-window ledger.

    Call ``begin_period`` immediately before dispatch, then ``complete_period``
    only after all actual observations and controller feedback are committed.
    A failed or in-flight transaction cannot be crossed by a new reference.
    The caller still owns and verifies the actual physical operation.
    """

    def __init__(self, *, max_periods, **registration):
        self.max_periods = _period_budget(max_periods)
        super().__init__(**registration)
        self._inflight = None
        self.early_refreshes = []

    @property
    def inflight(self):
        """Return the pre-dispatch snapshot without exposing mutable state."""
        return copy.deepcopy(self._inflight)

    def begin_period(self, *, before):
        try:
            if (
                self.phase != "executing"
                or self._inflight is not None
                or self.segment_periods >= 8
                or self.completed_periods >= self.max_periods
                or _snapshot(before) != self._current
            ):
                raise ValueError("A current accepted segment and idle, in-budget dispatch are required")
            self._inflight = self.current
        except Exception as error:
            self._fail(error)

    def complete_period(self, *, before, after, inner_samples):
        try:
            if self._inflight is None or _snapshot(before) != self._inflight:
                raise ValueError("A matching in-flight action is required before completing a period")
            super().complete_period(before=before, after=after, inner_samples=inner_samples)
            self._inflight = None
            if self.completed_periods == self.max_periods:
                self.phase = "done"
        except Exception as error:
            self._fail(error)

    def refresh_early(self, *, trigger):
        """Close a 1–7 period segment after a caller-verified current trigger.

        The returned Boolean reports whether another request remains in budget.
        It does not independently verify why the owner requested a refresh.
        Full eight-period segments close automatically in ``complete_period``.
        """
        try:
            if (
                self.phase != "executing"
                or self._inflight is not None
                or not 1 <= self.segment_periods < 8
                or type(trigger) is not dict
                or trigger.get("replan") is not True
                or type(trigger.get("current_step")) is not int
                or trigger["current_step"] != self._current["step"]
                or type(trigger.get("reason")) is not str
                or not trigger["reason"]
            ):
                raise ValueError("An idle, completed segment and current positive trigger are required")
            self.early_refreshes.append(
                dict(
                    start_step=self._current["step"] - self.segment_periods,
                    end_step=self._current["step"],
                    completed_periods=self.segment_periods,
                    trigger=copy.deepcopy(trigger),
                )
            )
            remaining = self.issued_requests < self.max_requests and self.completed_periods < self.max_periods
            self.phase = "ready" if remaining else "done"
            return remaining
        except Exception as error:
            self._fail(error)


class AdaptiveModelAdmission(CausalModelAdmission):
    """Independently admit current images at variable 1–8 period intervals.

    The service verifies sequencing and image identity, not whether a simulator
    really performed the periods claimed by the request owner.
    """

    def __init__(self, *, max_periods, **registration):
        self.max_periods = _period_budget(max_periods)
        super().__init__(**registration)
        self._last_request = None

    @property
    def last_request_snapshot(self):
        return copy.deepcopy(self._last_request)

    def admit(self, request):
        try:
            if self.phase != "ready" or self.admitted_requests >= self.max_requests:
                raise ValueError("One service request at a time within budget is required")
            _validate_request(request)
            current = {key: request[key] for key in SNAPSHOT_KEYS}
            first = self.initial["step"]
            if self._last_request is None:
                cadence_valid = current == self.initial
            else:
                delta = current["step"] - self._last_request["step"]
                cadence_valid = 1 <= delta <= 8
            if (
                request["session"] != self.session
                or request["request_id"] != self.admitted_requests
                or not cadence_valid
                or current["step"] >= first + self.max_periods
                or not math.isclose(
                    current["sim_time_s"],
                    self.initial["sim_time_s"] + 0.05 * (current["step"] - first),
                    rel_tol=0,
                    abs_tol=1e-9,
                )
                or request["task"] != self.task
                or request["seed"] != self.seed
                or request["model_sha256"] != self.model_sha256
                or request["denoise_steps"] != self.denoise_steps
            ):
                raise ValueError("Service request differs from the registered adaptive sequence and sampling")
            inputs = model_inputs(request, image_root=self.image_root)
            self._last_request = dict(current)
            self.admitted_requests += 1
            self.phase = "pending"
            return inputs
        except Exception:
            self.phase = "failed"
            raise


def check_adaptive_stop(service, frame):
    """Validate a stop after a completed response and within its current window."""
    if type(service) is not AdaptiveModelAdmission or type(frame) is not dict:
        raise ValueError("An adaptive model service and explicit stop frame are required")
    latest = service.last_request_snapshot
    lower = service.initial["step"] if latest is None else latest["step"]
    upper = lower if latest is None else min(lower + 8, service.initial["step"] + service.max_periods)
    if (
        service.phase != "ready"
        or service.admitted_requests >= service.max_requests
        or set(frame) != {"version", "kind", "session", "next_step"}
        or type(frame["version"]) is not int
        or frame["version"] != 1
        or frame["kind"] != "stop"
        or frame["session"] != service.session
        or type(frame["next_step"]) is not int
        or not lower <= frame["next_step"] <= upper
    ):
        raise ValueError("Stop frame differs from the waiting adaptive stream")
    return dict(frame)
