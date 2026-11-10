"""Use two independently audited pose points in a causal feedback transaction."""

import copy

from s2a.research.joint_control.preview.feasible_start_restoration_feedback import FeasibleStartRestorationSolverAudit
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search.two_step_viability import solve_two_step_viability


class TwoStepFeedbackSolverAudit(FeasibleStartRestorationSolverAudit):
    """Retain existing transaction and safety checks while enforcing two poses."""

    def __init__(self, receipt, *, solver=solve_two_step_viability, auditor=audit_two_step_witness):
        if not isinstance(receipt, dict) or "two_step_feedback_traces" in receipt:
            raise ValueError("One fresh two-step feedback receipt is required")
        self.two_step_traces = receipt["two_step_feedback_traces"] = []

        def recorded_solver(*args, feasibility_restoration, **kwargs):
            if feasibility_restoration is not True:
                raise ValueError("Original safety restoration must remain enabled")
            try:
                solved = solver(*args, **kwargs)
            except (AssertionError, ValueError, RuntimeError) as error:
                self.two_step_traces.append(dict(solver_raised={"type": type(error).__name__, "message": str(error)}))
                raise
            phases, count, info = (
                solved.get("search_phases"),
                solved.get("response_evaluations"),
                solved.get("initialization"),
            )
            if (
                not isinstance(phases, list)
                or not isinstance(count, int)
                or not 0 <= count <= 320
                or not isinstance(info, dict)
            ):
                raise ValueError("Two-step search must account for the shared model budget")
            names = [phase.get("phase") for phase in phases]
            if names not in ([], ["two_point_feasibility"], ["two_point_feasibility", "future_quality"]):
                raise ValueError("Two-step search phase ordering is invalid")
            if phases:
                first = phases[0].get("iterations")
                if first is not None and (not isinstance(first, int) or not 0 <= first <= 60):
                    raise ValueError("Two-point feasibility exceeds its iteration budget")
                if len(phases) == 2:
                    second = phases[1].get("iterations")
                    if first is None or (
                        second is not None and (not isinstance(second, int) or not 0 <= second <= 120 - first)
                    ):
                        raise ValueError("Two-step search exceeds its shared iteration budget")
            info.update(actual_model_rollouts=count, seed_rollout_reused=True, solver_lp_attempted=False)
            candidate = solved.get("best_joint_safe")
            self.two_step_traces.append(
                dict(
                    status=solved.get("status"),
                    response_evaluations=count,
                    phases=copy.deepcopy(phases),
                    joint_safe_evaluations=solved.get("joint_safe_evaluations"),
                    prefix_hard_evaluations=solved.get("prefix_hard_evaluations"),
                    closest_prefix_rejected=copy.deepcopy(solved.get("closest_prefix_rejected")),
                    candidate_prefix_errors=None if candidate is None else candidate["tcp_errors"][:2].copy(),
                    candidate_future_quality_slack=None if candidate is None else candidate["future_quality_slack"],
                    audit_status="not_attempted" if candidate is None else "pending",
                )
            )
            return solved

        def checked_auditor(*args, enforce_pose_caps, **kwargs):
            if enforce_pose_caps is not False:
                raise ValueError("Only the explicitly audited two-point prefix is required")
            return auditor(*args, **kwargs)

        super().__init__(receipt, solver=recorded_solver, auditor=checked_auditor)

    def audit(self, *args, **kwargs):
        witness = args[0] if args else kwargs.get("witness")
        current = self._pending is not None and witness is self._pending[0]
        try:
            verified = super().audit(*args, **kwargs)
        except (AssertionError, ValueError, RuntimeError) as error:
            if current:
                self.two_step_traces[-1]["audit_status"] = "rejected"
                self.two_step_traces[-1]["audit_error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        self.two_step_traces[-1]["audit_status"] = "passed"
        return verified
