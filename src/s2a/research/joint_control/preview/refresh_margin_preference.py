"""Enable cached-candidate preference only after an audited physical prefix."""

from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from s2a.research.trajectory_search import margin_guarded_fallback


class RefreshMarginPreference:
    """Keep the original solver through step 15, then prefer interior candidates."""

    def __init__(self):
        self.enabled = False
        self.adapter = None

    def solve(self, *args, **kwargs):
        return margin_guarded_fallback.solve_margin_guarded_fallback(*args, prefer_refresh=self.enabled, **kwargs)

    def adapter_factory(self, receipt):
        if self.adapter is not None or self.enabled:
            raise ValueError("One isolated adapter is required per physical episode")
        self.adapter = SharedShiftFallbackAudit(receipt, solver=self.solve)
        return self.adapter

    def activate(self, policy, owner, receipt):
        """Activate only at the completed step-16 boundary, with no pending action."""
        arm, inner = policy.arm_policy, policy.inner_history
        tracking = arm.tracking
        refiner = tracking._refiner
        past = inner.snapshot()
        if (
            self.enabled
            or self.adapter is None
            or type(receipt) is not dict
            or "refresh_preference_activation" in receipt
            or receipt.get("control_steps_completed") != 16
            or policy.start_step != 8
            or policy.max_steps != 8
            or policy.pending is not None
            or policy.last_inner_loss
            or getattr(inner, "_owner", None) is not owner
            or past.get("completed_periods") != 8
            or past.get("last_completed_step") != 15
            or past.get("future_guarantee") is not False
            or arm.pending is not None
            or arm.coupled.pending is not None
            or arm.coupled.completed_periods != 16
            or tracking.phase != "ready"
            or tracking.history.completed_periods != 16
            or tracking.history.pending is not None
            or tracking.history.pending_plan is not None
            or refiner.history is not inner
            or getattr(refiner.solver, "__self__", None) is not self.adapter
            or self.adapter._pending is not None
        ):
            raise ValueError("The exact completed and idle step-16 feedback history is required")
        receipt["refresh_preference_activation"] = dict(step=16, completed_periods=16, future_observations=0)
        self.enabled = True
