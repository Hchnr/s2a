"""Dispatch margin-first or original two-point candidates only after audit."""

import copy

from s2a.research.joint_control.preview.feasible_start_restoration_feedback import FeasibleStartRestorationSolverAudit
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import margin_guarded_fallback as policy
from s2a.research.trajectory_search import margin_two_step_search as margin


class MarginGuardedFeedbackAudit(FeasibleStartRestorationSolverAudit):
    """Preserve the shared budgets and independently audit the selected scope."""

    def __init__(self, receipt, *, solver=policy.solve_margin_guarded_fallback, auditor=audit_two_step_witness):
        if not isinstance(receipt, dict) or "margin_guarded_traces" in receipt:
            raise ValueError("One fresh shared-search receipt is required")
        self.shared_traces = receipt["margin_guarded_traces"] = []

        def recorded_solver(*args, feasibility_restoration, **kwargs):
            if feasibility_restoration is not True:
                raise ValueError("Original safety restoration must remain enabled")
            solved = solver(*args, **kwargs)
            trace = solved.get("shared_search")
            count = solved.get("response_evaluations")
            candidate = solved.get("best_joint_safe")
            info = solved.get("initialization")
            if (
                not isinstance(trace, dict)
                or not isinstance(info, dict)
                or type(count) is not int
                or not 0 <= count <= policy.MAX_RESPONSES
                or trace.get("response_budget") != policy.MAX_RESPONSES
                or trace.get("feasibility_iteration_budget") != policy.MAX_FEASIBILITY_ITERATIONS
                or trace.get("total_iteration_budget") != policy.MAX_TOTAL_ITERATIONS
                or trace.get("start_lp_budget") != 2
                or type(trace.get("primary_responses")) is not int
                or type(trace.get("fallback_responses")) is not int
                or min(trace["primary_responses"], trace["fallback_responses"]) < 0
                or trace["primary_responses"] + trace["fallback_responses"] != count
                or trace.get("fallback_lp_attempts") != 0
                or not 0 <= info.get("start_lp_attempts", -1) <= 2
                or info.get("actual_model_rollouts") != count
            ):
                raise ValueError("Shared response, iteration and LP accounting must remain exact")
            scope = trace.get("selected_scope")
            if scope not in (None, "margin_two_step", "guarded_two_step") or (scope is None) != (candidate is None):
                raise ValueError("Selected scope must match the candidate")
            if candidate is not None and (
                solved.get("status") != "margin_guarded_candidate"
                or candidate.get("selection_scope") != scope
                or candidate.get("hard_pose_prefix") != 2
                or candidate.get("prefix_selection_hard") is not True
                or (scope == "margin_two_step" and not margin.margin_prefix_satisfied(candidate["tcp_errors"]))
                or (scope == "guarded_two_step" and trace.get("fallback_initializer_reused") is not True)
            ):
                raise ValueError("Selected candidates must satisfy their model selection contract")
            if scope == "margin_two_step" and (
                trace["fallback_responses"] != 0
                or trace.get("primary_status") != "two_step_margin_candidate"
                or trace.get("fallback_status") != "not_attempted"
            ):
                raise ValueError("A margin candidate must retain priority over fallback")
            if scope == "guarded_two_step" and (
                trace.get("primary_status") != "two_step_margin_no_witness"
                or trace.get("fallback_status") != "two_step_candidate"
            ):
                raise ValueError("Guarded fallback requires a completed primary search without a margin candidate")
            primary_first, primary_total = (
                trace.get("primary_feasibility_iterations"),
                trace.get("primary_total_iterations"),
            )
            calls = trace.get("optimizer_calls")
            if not isinstance(calls, list) or len(calls) > 2:
                raise ValueError("At most two fallback optimizer phases are allowed")
            if primary_first is None or primary_total is None:
                if calls or trace["fallback_responses"] or scope == "guarded_two_step":
                    raise ValueError("Unknown primary iterations forbid fallback")
            else:
                if (
                    type(primary_first) is not int
                    or type(primary_total) is not int
                    or not 0 <= primary_first <= policy.MAX_FEASIBILITY_ITERATIONS
                    or not primary_first <= primary_total <= policy.MAX_TOTAL_ITERATIONS
                ):
                    raise ValueError("Primary iteration accounting must remain within budget")
                used = 0
                for index, call in enumerate(calls):
                    if call.get("phase") != ("two_point_feasibility" if index == 0 else "future_quality"):
                        raise ValueError("Fallback optimizer phases must remain in order")
                    limit, iterations = call.get("limit"), call.get("iterations")
                    maximum = min(
                        policy.MAX_TOTAL_ITERATIONS - primary_total - used,
                        policy.MAX_FEASIBILITY_ITERATIONS - primary_first
                        if index == 0
                        else policy.MAX_TOTAL_ITERATIONS,
                    )
                    if (
                        type(limit) is not int
                        or not 0 < limit <= maximum
                        or (iterations is not None and (type(iterations) is not int or not 0 <= iterations <= limit))
                    ):
                        raise ValueError("Fallback optimization exceeded the shared iteration budget")
                    if iterations is None:
                        if candidate is not None or index != len(calls) - 1:
                            raise ValueError("An interrupted optimization cannot supply a candidate")
                    else:
                        used += iterations
            info["solver_lp_attempted"] = False
            self.shared_traces.append(
                dict(
                    **copy.deepcopy(trace),
                    status=solved.get("status"),
                    response_evaluations=count,
                    audit_status="not_attempted" if candidate is None else "pending",
                )
            )
            return solved

        def checked_auditor(witness, *args, enforce_pose_caps, **kwargs):
            if enforce_pose_caps is not False:
                raise ValueError("Only the independently audited two-point prefix is required")
            verified = auditor(witness, *args, **kwargs)
            scope = witness.get("selection_scope")
            if scope not in ("margin_two_step", "guarded_two_step"):
                raise ValueError("A known two-point selection scope is required")
            if scope == "margin_two_step" and not margin.margin_prefix_satisfied(verified["tcp_errors"]):
                raise ValueError("Independent reconstruction must satisfy the fixed model margins")
            verified.update(selection_scope=scope, model_margin_enforced=scope == "margin_two_step")
            return verified

        super().__init__(receipt, solver=recorded_solver, auditor=checked_auditor)

    def audit(self, *args, **kwargs):
        witness = args[0] if args else kwargs.get("witness")
        current = self._pending is not None and witness is self._pending[0]
        try:
            verified = super().audit(*args, **kwargs)
        except (AssertionError, ValueError, RuntimeError) as error:
            if current:
                self.shared_traces[-1]["audit_status"] = "rejected"
                self.shared_traces[-1]["audit_error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        verified["selection_scope"] = witness["selection_scope"]
        self.shared_traces[-1]["audit_status"] = "passed"
        self.shared_traces[-1]["independent_prefix_errors"] = verified["tcp_errors"][:2].copy()
        return verified
