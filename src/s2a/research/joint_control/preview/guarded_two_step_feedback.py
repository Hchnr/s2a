"""Record scaled search while preserving the audited two-point transaction."""

from s2a.research.joint_control.preview.two_step_feedback import TwoStepFeedbackSolverAudit
from s2a.research.trajectory_search.guarded_two_step_search import OBJECTIVE_SCALE, SEARCH_GUARD, solve_guarded_two_step


class GuardedTwoStepFeedbackSolverAudit(TwoStepFeedbackSolverAudit):
    """Retain the original two-point acceptance and independent safety audit."""

    def __init__(self, receipt, *, solver=solve_guarded_two_step):
        if not isinstance(receipt, dict) or "guarded_two_step_traces" in receipt:
            raise ValueError("One fresh guarded-search receipt is required")
        self.guard_traces = receipt["guarded_two_step_traces"] = []

        def recorded_solver(*args, **kwargs):
            solved = solver(*args, **kwargs)
            details = solved.get("first_stage_search_objective")
            if details != dict(
                normalized_guard=SEARCH_GUARD,
                scale=OBJECTIVE_SCALE,
                acceptance_caps_changed=False,
                future_stage_changed=False,
            ):
                raise ValueError("The guarded first-stage objective and unchanged acceptance must be recorded")
            self.guard_traces.append(dict(**details, response_evaluations=solved.get("response_evaluations")))
            return solved

        super().__init__(receipt, solver=recorded_solver)
