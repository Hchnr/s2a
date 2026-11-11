"""Use a completed shifted plan only after shared-budget dual search stops."""

import copy
from types import FunctionType

from s2a.research.joint_control.preview.causal_shift_fallback import CausalShiftFallbackAudit
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import margin_guarded_fallback as shared
from s2a.research.trajectory_search import margin_two_step_search as margin


def validate_shared_result(solved):
    """Require the original joint budgets and distinguish both dual scopes."""
    trace, info, count = solved.get("shared_search"), solved.get("initialization"), solved.get("response_evaluations")
    if (
        not isinstance(trace, dict)
        or not isinstance(info, dict)
        or type(count) is not int
        or not 0 <= count <= shared.MAX_RESPONSES
        or trace.get("response_budget") != shared.MAX_RESPONSES
        or trace.get("feasibility_iteration_budget") != shared.MAX_FEASIBILITY_ITERATIONS
        or trace.get("total_iteration_budget") != shared.MAX_TOTAL_ITERATIONS
        or trace.get("start_lp_budget") != 2
        or type(trace.get("primary_responses")) is not int
        or type(trace.get("fallback_responses")) is not int
        or min(trace["primary_responses"], trace["fallback_responses"]) < 0
        or trace["primary_responses"] + trace["fallback_responses"] != count
        or trace.get("fallback_lp_attempts") != 0
        or type(info.get("start_lp_attempts")) is not int
        or not 0 <= info["start_lp_attempts"] <= 2
        or info.get("actual_model_rollouts") != count
    ):
        raise ValueError("The shared response, iteration and LP accounting must remain exact")
    scope, candidate = trace.get("selected_scope"), solved.get("best_joint_safe")
    if scope not in (None, "margin_two_step", "guarded_two_step") or (scope is None) != (candidate is None):
        raise ValueError("The shared selection scope must match its candidate")
    if candidate is not None and (
        candidate.get("witness_found") is not True
        or candidate.get("selection_scope") != scope
        or candidate.get("hard_pose_prefix") != 2
        or candidate.get("prefix_selection_hard") is not True
        or (scope == "margin_two_step" and not margin.margin_prefix_satisfied(candidate["tcp_errors"]))
        or (scope == "guarded_two_step" and trace.get("fallback_initializer_reused") is not True)
    ):
        raise ValueError("A selected candidate must satisfy its original dual-pose scope")
    if scope == "margin_two_step" and (
        trace.get("primary_status") != "two_step_margin_candidate"
        or trace.get("fallback_status") != "not_attempted"
        or trace["fallback_responses"] != 0
    ):
        raise ValueError("The margin candidate must keep priority")
    if scope == "guarded_two_step" and (
        trace.get("primary_status") != "two_step_margin_no_witness"
        or trace.get("fallback_status") != "two_step_candidate"
    ):
        raise ValueError("Guarded fallback requires a completed margin search without a candidate")
    primary = shared.primary_iterations(trace.get("primary_phases"))
    if primary is None:
        if trace.get("primary_feasibility_iterations") is not None or trace.get("primary_total_iterations") is not None:
            raise ValueError("Unknown primary iterations must remain unknown")
    elif (trace.get("primary_feasibility_iterations"), trace.get("primary_total_iterations")) != primary:
        raise ValueError("Primary iteration totals must match the recorded phases")
    calls = trace.get("optimizer_calls")
    fallback_phases = trace.get("fallback_phases")
    if not isinstance(calls, list) or len(calls) > 2 or not isinstance(fallback_phases, list):
        raise ValueError("At most two guarded optimizer phases are allowed")
    if trace.get("fallback_status") in ("two_step_candidate", "two_step_no_witness") and (
        trace.get("fallback_initializer_reused") is not True
        or trace["fallback_responses"] < 1
        or not calls
        or len(fallback_phases) != len(calls)
        or any(
            phase.get("phase") != call.get("phase") or phase.get("iterations") != call.get("iterations")
            for phase, call in zip(fallback_phases, calls, strict=True)
        )
    ):
        raise ValueError("A completed guarded result must retain its reused initializer and optimizer phases")
    if primary is None:
        if calls or trace["fallback_responses"] or scope == "guarded_two_step":
            raise ValueError("Unknown primary iterations forbid guarded fallback")
    else:
        used = 0
        for index, call in enumerate(calls):
            if call.get("phase") != ("two_point_feasibility" if index == 0 else "future_quality"):
                raise ValueError("Guarded optimizer phases must remain in order")
            limit, iterations = call.get("limit"), call.get("iterations")
            remaining = min(
                shared.MAX_TOTAL_ITERATIONS - primary[1] - used,
                shared.MAX_FEASIBILITY_ITERATIONS - primary[0] if index == 0 else shared.MAX_TOTAL_ITERATIONS,
            )
            if (
                type(limit) is not int
                or not 0 < limit <= remaining
                or (iterations is not None and (type(iterations) is not int or not 0 <= iterations <= limit))
            ):
                raise ValueError("Guarded optimization exceeded the shared iteration budget")
            if iterations is None:
                if candidate is not None or index != len(calls) - 1:
                    raise ValueError("An interrupted optimizer cannot supply a candidate")
            else:
                used += iterations
    return count


class SharedShiftFallbackAudit(CausalShiftFallbackAudit):
    """Audit margin, guarded and completed-plan selections in that order."""

    def __init__(
        self,
        receipt,
        *,
        solver=shared.solve_margin_guarded_fallback,
        dual_auditor=audit_two_step_witness,
        first_auditor=audit_full_horizon_witness,
    ):
        if not isinstance(receipt, dict) or "shared_shift_traces" in receipt:
            raise ValueError("One fresh shared-shift receipt is required")
        self.shared_traces = receipt["shared_shift_traces"] = []

        def recorded_solver(*args, **kwargs):
            solved = solver(*args, **kwargs)
            count = validate_shared_result(solved)
            trace, candidate = solved["shared_search"], solved.get("best_joint_safe")
            original_status = solved.get("status")
            if candidate is not None:
                if original_status != "margin_guarded_candidate":
                    raise ValueError("A shared dual candidate must retain its original status")
                normalized = "two_step_candidate"
            elif original_status != "margin_guarded_no_witness":
                raise ValueError("A missing shared candidate must retain its original status")
            elif (
                trace["primary_status"] == "two_step_margin_no_witness"
                and trace["fallback_status"] == "two_step_no_witness"
                and trace["primary_feasibility_iterations"] is not None
                and all(call["iterations"] is not None for call in trace["optimizer_calls"])
            ):
                normalized = "two_step_no_witness"
            else:
                normalized = "shared_search_ineligible"
            self.shared_traces.append(
                dict(
                    shared_search=copy.deepcopy(trace),
                    shared_status=original_status,
                    primary_response_evaluations=count,
                    normalized_status=normalized,
                )
            )
            return dict(solved, status=normalized)

        def scoped_dual_auditor(witness, *args, **kwargs):
            verified = dual_auditor(witness, *args, **kwargs)
            scope = witness.get("selection_scope")
            if scope not in ("margin_two_step", "guarded_two_step"):
                raise ValueError("A known independently audited dual-pose scope is required")
            if scope == "margin_two_step" and not margin.margin_prefix_satisfied(verified["tcp_errors"]):
                raise ValueError("Independent reconstruction must satisfy the fixed model margins")
            verified.update(selection_scope=scope, model_margin_enforced=scope == "margin_two_step")
            return verified

        super().__init__(receipt, solver=recorded_solver, dual_auditor=scoped_dual_auditor, first_auditor=first_auditor)

    @staticmethod
    def _validate_primary(solved):
        return validate_shared_result(solved)

    def solve(self, *args, **kwargs):
        """Count an optional shifted rollout in both the receipt and callback."""
        callback = kwargs.get("on_response")
        namespace = CausalShiftFallbackAudit.solve.__globals__.copy()
        original_rollout = namespace["rollout_joint_preview"]

        def observed_shifted_rollout(*rollout_args, **rollout_kwargs):
            if callback is not None:
                callback("attempted")
            result = original_rollout(*rollout_args, **rollout_kwargs)
            if callback is not None:
                callback("completed")
            return result

        namespace["rollout_joint_preview"] = observed_shifted_rollout
        old = CausalShiftFallbackAudit.solve
        function = FunctionType(old.__code__, namespace, old.__name__, old.__defaults__, old.__closure__)
        function.__kwdefaults__ = dict(old.__kwdefaults__ or {})
        result = function(self, *args, **kwargs)
        trace, prior = self.traces[-1], self.shared_traces[-1]
        selected = trace["selected_scope"]
        scope = prior["shared_search"]["selected_scope"] if selected == "two_step" else selected
        if selected == "first_only":
            scope = "shifted_plan_first_pose"
            result["selection_scope"] = scope
        elif selected == "two_step":
            result["status"] = f"{scope}_candidate"
        elif result.get("status") == "two_step_no_witness":
            result["status"] = "shared_shift_no_witness"
        prior.update(
            selected_scope=scope,
            shifted_status=trace["fallback_status"],
            shifted_rollouts=trace["fallback_rollouts"],
            response_evaluations=trace["response_evaluations"],
            audit_status=trace["audit_status"],
        )
        return result

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
        self.shared_traces[-1]["audit_status"] = "passed"
        self.shared_traces[-1]["independent_prefix_errors"] = verified["tcp_errors"][:2].copy()
        return verified
