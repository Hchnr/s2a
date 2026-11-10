"""Record first-hard restoration candidates and their independent audit."""

import numpy as np

from s2a.research.joint_control.preview.restoration_horizon_feedback import RestorationHorizonSolverAudit
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.first_hard_restoration import solve_first_hard_restoration


class FirstHardRestorationSolverAudit(RestorationHorizonSolverAudit):
    """Use the existing transaction audit while preserving decision evidence."""

    def __init__(self, receipt, *, solver=solve_first_hard_restoration, auditor=audit_full_horizon_witness):
        if not isinstance(receipt, dict) or "first_hard_restoration_traces" in receipt:
            raise ValueError("One fresh decision receipt is required")
        self.traces = receipt["first_hard_restoration_traces"] = []

        def recorded_solver(*args, **kwargs):
            try:
                solved = solver(*args, **kwargs)
            except (AssertionError, ValueError, RuntimeError) as error:
                self.traces.append(dict(solver_raised={"type": type(error).__name__, "message": str(error)}))
                raise
            keys = (
                "status",
                "response_evaluations",
                "iterations",
                "optimizer_success",
                "optimizer_message",
                "optimizer_future_slack",
                "joint_safe_evaluations",
                "first_hard_evaluations",
                "closest_first_rejected",
            )
            trace = {key: solved[key] for key in keys if key in solved}
            candidate = solved.get("best_joint_safe")
            trace["candidate"] = (
                None
                if candidate is None
                else {
                    key: np.asarray(candidate[key]).copy()
                    if key in ("planned_commands", "first_tcp_error", "tcp_errors", "terminal_normals")
                    else candidate[key]
                    for key in (
                        "planned_commands",
                        "first_tcp_error",
                        "tcp_errors",
                        "terminal_normals",
                        "quality_slack",
                        "future_quality_slack",
                        "first_pose_optimized_hard",
                    )
                    if key in candidate
                }
            )
            trace["audit_status"] = "not_attempted" if candidate is None else "pending"
            self.traces.append(trace)
            return solved

        def recorded_auditor(*args, **kwargs):
            try:
                verified = auditor(*args, **kwargs)
            except (AssertionError, ValueError, RuntimeError) as error:
                self.traces[-1]["audit_detail"] = {"raised": type(error).__name__, "message": str(error)}
                raise
            keys = (
                "passed",
                "pose_horizon",
                "first_pose_caps_satisfied",
                "first_pose_caps_enforced",
                "all_pose_caps_satisfied",
                "all_pose_caps_enforced",
                "braking_constraint_satisfied",
                "braking_constraint_enforced",
                "constraint_minima",
                "first_tcp_error",
                "tcp_errors",
                "independent_endpoint_queries",
                "independent_transition_steps",
            )
            self.traces[-1]["audit_detail"] = {key: verified[key] for key in keys if key in verified}
            return verified

        super().__init__(solver=recorded_solver, auditor=recorded_auditor)

    def audit(self, *args, **kwargs):
        witness = args[0] if args else kwargs.get("witness")
        current = self._pending is not None and witness is self._pending[0]
        try:
            verified = super().audit(*args, **kwargs)
        except (AssertionError, ValueError, RuntimeError) as error:
            if current:
                self.traces[-1]["audit_status"] = "rejected"
                self.traces[-1]["audit_error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        self.traces[-1]["audit_status"] = "passed"
        return verified
