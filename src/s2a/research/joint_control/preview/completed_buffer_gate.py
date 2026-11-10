"""Stop before dispatch after a measured current or completed inner buffer loss."""

import numpy as np


class CompletedBufferGate:
    """Track only actual completed inner samples, never a proposed trajectory."""

    def __init__(self, *, initial_inner_loss):
        if type(initial_inner_loss) is not bool:
            raise ValueError("Explicit previous inner-buffer status required")
        self.last_inner_loss = initial_inner_loss
        self.pending = None

    def select(self, q, limits, jacobian, previous, offset, choose):
        q, limits = np.asarray(q, dtype=float), np.asarray(limits, dtype=float)
        if (
            self.pending is not None
            or q.shape != (7,)
            or limits.shape != (7, 2)
            or not np.isfinite(q).all()
            or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
            or type(offset) is not int
            or offset < 0
        ):
            raise ValueError("One finite current state and no pending dispatch required")
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        if margin < 0.05 or self.last_inner_loss:
            return {"stop": {"reason": "observed_joint_buffer_loss", "minimum_current_margin_rad": margin}}
        decision = choose(q, limits, jacobian, previous, offset)
        if not isinstance(decision, dict):
            raise ValueError("Candidate must produce a dispatch or explicit stop")
        if "stop" not in decision:
            self.pending = {"offset": offset, "limits": limits.copy()}
        return decision

    def complete(self, offset, actual_inner):
        if self.pending is None or offset != self.pending["offset"]:
            raise ValueError("Only the staged physical period may complete")
        q = np.asarray(actual_inner, dtype=float)
        if q.shape != (25, 7) or not np.isfinite(q).all():
            raise ValueError("All 25 actual completed joint samples required")
        limits = self.pending["limits"]
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        self.last_inner_loss = margin < 0.05
        self.pending = None
        return {"minimum_inner_margin_rad": margin, "below_buffer": self.last_inner_loss}
