"""Require independently reconstructed model margins before feedback dispatch."""

import copy

from s2a.research.joint_control.preview.two_step_feedback import TwoStepFeedbackSolverAudit
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import margin_two_step_search as margin


class MarginTwoStepFeedbackSolverAudit(TwoStepFeedbackSolverAudit):
    """Keep the original safety audit and add the registered two-pose margins."""

    def __init__(self, receipt, *, solver=margin.solve_margin_two_step, auditor=audit_two_step_witness):
        if not isinstance(receipt, dict) or "margin_two_step_traces" in receipt:
            raise ValueError("One fresh model-margin receipt is required")
        self.margin_traces = receipt["margin_two_step_traces"] = []

        def recorded_solver(*args, **kwargs):
            solved = solver(*args, **kwargs)
            details = solved.get("margin_search")
            count = solved.get("response_evaluations")
            candidate = solved.get("best_joint_safe")
            if (
                not isinstance(details, dict)
                or details.get("model_position_margin_m") != margin.POSITION_MARGIN_M
                or details.get("model_rotation_margin_rad") != margin.ROTATION_MARGIN_RAD
                or details.get("selection_tolerance") != margin.SELECTION_TOLERANCE
                or details.get("objective_scale") != 10000.0
                or details.get("original_acceptance_caps_changed") is not False
                or details.get("extra_model_responses") != 0
                or details.get("recorded_rollout_calls") != count
                or not isinstance(count, int)
                or not 0 <= count <= 320
                or not isinstance(details.get("margin_prefix_evaluations"), int)
                or details["margin_prefix_evaluations"] < 0
                or (candidate is None) != (details["margin_prefix_evaluations"] == 0)
            ):
                raise ValueError("Exact model margins and shared response accounting are required")
            if candidate is not None and not margin.margin_prefix_satisfied(candidate["tcp_errors"]):
                raise ValueError("A selected model candidate must satisfy both registered margins")
            self.margin_traces.append(
                dict(
                    **copy.deepcopy(details),
                    status=solved.get("status"),
                    response_evaluations=count,
                    audit_status="not_attempted" if candidate is None else "pending",
                )
            )
            return solved

        def checked_auditor(*args, **kwargs):
            verified = auditor(*args, **kwargs)
            if not margin.margin_prefix_satisfied(verified["tcp_errors"]):
                raise ValueError("Independent two-point reconstruction must satisfy the fixed model margins")
            verified.update(
                model_margin_satisfied=True,
                model_position_margin_m=margin.POSITION_MARGIN_M,
                model_rotation_margin_rad=margin.ROTATION_MARGIN_RAD,
            )
            return verified

        super().__init__(receipt, solver=recorded_solver, auditor=checked_auditor)

    def audit(self, *args, **kwargs):
        witness = args[0] if args else kwargs.get("witness")
        current = self._pending is not None and witness is self._pending[0]
        try:
            verified = super().audit(*args, **kwargs)
        except (AssertionError, ValueError, RuntimeError) as error:
            if current:
                self.margin_traces[-1]["audit_status"] = "rejected"
                self.margin_traces[-1]["audit_error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        self.margin_traces[-1]["audit_status"] = "passed"
        self.margin_traces[-1]["independent_prefix_errors"] = verified["tcp_errors"][:2].copy()
        return verified
