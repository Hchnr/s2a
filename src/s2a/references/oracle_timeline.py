"""Build a time-aligned oracle reference without exposing recorded actions."""

import numpy as np

from s2a.geometry import validate_transform

HORIZON = 40
STRIDE = 8
ALIGNMENT_ATOL = 1e-10


class OracleTimeline:
    """Join overlapping correct-reference windows and expose bounded H40 slices."""

    def __init__(self, windows, *, periods):
        if type(periods) is not int or periods < 1 or not isinstance(windows, list):
            raise ValueError("A positive period count and ordered reference windows are required")
        if len(windows) != (periods + STRIDE - 1) // STRIDE:
            raise ValueError("Correct-reference windows must cover the registered timeline exactly")
        poses, openings = [], []
        prior = None
        maximum_pose_difference = 0.0
        for index, row in enumerate(windows):
            if not isinstance(row, dict) or row.get("step") != index * STRIDE or type(row.get("step")) is not int:
                raise ValueError("Correct-reference windows must start at consecutive eight-step boundaries")
            target = validate_transform(np.asarray(row["effective_targets_tcp"], dtype=float)).copy()
            opening = np.asarray(row["openings"], dtype=float).copy()
            if (
                target.shape != (HORIZON, 4, 4)
                or opening.shape != (HORIZON,)
                or not np.isfinite(opening).all()
                or np.any((opening < 0) | (opening > 1))
            ):
                raise ValueError("Each correct reference must contain 40 finite poses and openings")
            if prior is not None:
                difference = float(np.max(np.abs(prior[0][STRIDE:] - target[:-STRIDE])))
                if difference > ALIGNMENT_ATOL or not np.array_equal(prior[1][STRIDE:], opening[:-STRIDE]):
                    raise ValueError("Overlapping correct references must describe the same future endpoints")
                maximum_pose_difference = max(maximum_pose_difference, difference)
            poses.append(target[:STRIDE])
            openings.append(opening[:STRIDE])
            prior = target, opening
        self._poses = np.concatenate(poses, axis=0)[:periods].copy()
        self._openings = np.concatenate(openings, axis=0)[:periods].copy()
        self._poses.setflags(write=False)
        self._openings.setflags(write=False)
        self.periods = periods
        self.window_count = len(windows)
        self.overlap_pose_max_difference = maximum_pose_difference

    def window(self, start):
        """Expose 40 desired endpoints, with explicit terminal-reference holds."""
        if type(start) is not int or not 0 <= start < self.periods:
            raise ValueError("The oracle cursor must name a remaining registered period")
        remaining = min(HORIZON, self.periods - start)
        indices = np.minimum(np.arange(start, start + HORIZON), self.periods - 1)
        return (
            self._poses[indices].copy(),
            self._openings[indices].copy(),
            dict(start_step=start, reference_endpoints=remaining, terminal_hold_endpoints=HORIZON - remaining),
        )
