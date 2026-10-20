"""Experimental reference recovery from an unreachable preceding chunk target.

The original governor remains frozen for its live comparison. This candidate
allows only an error-reducing chunk bridge to move while ordinary tracking
progress is paused. It preserves the preceding desired opening during that
bridge; observed loaded aperture is not inserted as a new opening command.
Generated model endpoints and their subsequent common clock are unchanged.
"""

import numpy as np

from s2a.geometry import pose_error
from s2a.reference_governor import ReferenceGovernor, interpolate_pose


class RecoveringReferenceGovernor(ReferenceGovernor):
    def set_reference(self, anchor, targets, openings, *, anchor_opening, model_dt):
        if not np.isscalar(anchor_opening) or not np.isfinite(anchor_opening) or not 0 <= anchor_opening <= 1:
            raise ValueError("measured anchor_opening must be in [0,1]")
        measured_opening = float(anchor_opening)
        has_previous = self._admitted is not None
        reference_opening = self._opening if has_previous else measured_opening
        summary = super().set_reference(anchor, targets, openings, anchor_opening=reference_opening, model_dt=model_dt)
        return dict(
            summary,
            measured_anchor_opening=measured_opening,
            opening_reference_at_anchor=float(reference_opening),
            bridge_opening_preserved=has_previous,
        )

    def advance(self, measured_tcp, dt):
        if self._poses is None:
            return super().advance(measured_tcp, dt)
        before_clock = self._clock
        before_target = self._admitted.copy()
        on_bridge = self._nominal[1] == 0 and before_clock < self._times[1]
        result = super().advance(measured_tcp, dt)
        result["nominal_tracking_scale"] = result["tracking_scale"]
        result["bridge_recovery_advanced"] = False
        if not on_bridge or result["tracking_scale"] >= 1:
            return result
        candidate_clock = min(before_clock + dt, self._times[1])
        if candidate_clock <= self._clock:
            return result
        fraction = candidate_clock / self._times[1]
        candidate = interpolate_pose(self._poses[0], self._poses[1], fraction)
        preceding_error, candidate_error = pose_error(before_target, measured_tcp), pose_error(candidate, measured_tcp)
        preceding = np.array([np.linalg.norm(preceding_error[:3]), np.linalg.norm(preceding_error[3:])])
        following = np.array([np.linalg.norm(candidate_error[:3]), np.linalg.norm(candidate_error[3:])])
        # Compare metres and radians separately, without a mixed-unit score.
        if not (np.all(following <= preceding + 1e-12) and np.any(following < preceding - 1e-12)):
            return result
        self._clock, self._admitted = float(candidate_clock), candidate
        self._opening = float((1 - fraction) * self._values[0] + fraction * self._values[1])
        result.update(
            target_tcp=candidate.copy(),
            opening=self._opening,
            model_time_s=0.0,
            retimed_clock_s=self._clock,
            tracking_scale=float((self._clock - before_clock) / dt),
            bridging=bool(self._clock < self._times[1]),
            complete=False,
            bridge_recovery_advanced=True,
        )
        return result
