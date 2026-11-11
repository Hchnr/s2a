"""Use a separately labeled bounded tracking fallback after strict candidates stop.

The fallback is independently audited against the original model-side joint,
rate, terminal, and braking constraints. It does not claim actual joint safety
or satisfaction of the original first-pose tracking thresholds.
"""

import numpy as np

from s2a.geometry import finite, pose_error
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain

POSITION_LIMIT = 0.02
ROTATION_LIMIT = 0.1
MAX_RESPONSES = 320
STATUS = "model_safe_tracking_degraded"


def _bounded_first(error):
    error = finite(error, (6,))
    return bool(
        np.linalg.norm(error[:3]) <= POSITION_LIMIT + 1e-10 and np.linalg.norm(error[3:]) <= ROTATION_LIMIT + 1e-10
    )


def _quality_gap(errors):
    errors = finite(errors, (8, 6))
    return max(
        0.0,
        float(
            np.max(
                np.r_[
                    np.linalg.norm(errors[:, :3], axis=1) / 0.01 - 1, np.linalg.norm(errors[:, 3:], axis=1) / 0.05 - 1
                ]
            )
        ),
    )


class ModelSafeTrackingFallback:
    """Preserve strict candidate priority and audit a past-only degraded fallback."""

    def __init__(self, receipt, *, primary=None, auditor=audit_full_horizon_witness):
        if not isinstance(receipt, dict) or "model_safe_tracking_fallback_traces" in receipt:
            raise ValueError("One fresh fallback receipt is required")
        self.traces = receipt["model_safe_tracking_fallback_traces"] = []
        self.primary = SharedShiftFallbackAudit(receipt) if primary is None else primary
        self.auditor = auditor
        self._pending = None
        self._history_attached = False

    def attach_history(self, history):
        if self._history_attached:
            raise ValueError("Exactly one branch-local history may be attached")
        self.primary.attach_history(history)
        self._history_attached = True

    def solve(self, model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs):
        if self._pending is not None or not self._history_attached:
            raise ValueError("The previous witness must be audited and branch history attached")
        seed, previous = finite(seed, (8, 6)), finite(previous, (6,))
        targets = finite(targets, (8, 4, 4)).copy()
        _ = finite(twists, (8, 6))
        solver_targets = targets.copy()
        solved = self.primary.solve(
            model, velocity, seed, previous, interface, site, tool, solver_targets, twists, lower, upper, **kwargs
        )
        if not np.array_equal(targets, solver_targets):
            raise ValueError("The strict search must preserve the generated targets")
        count = solved.get("response_evaluations")
        if type(count) is not int or not 0 <= count <= MAX_RESPONSES:
            raise ValueError("The shared response budget must remain bounded")
        strict_trace = self.primary.traces[-1]
        if strict_trace.get("response_evaluations") != count:
            raise ValueError("The strict search and fallback trace must account for the same responses")
        trace = {
            "strict_status": solved.get("status"),
            "strict_responses": count,
            "extra_rollouts": 0,
            "response_evaluations": count,
            "selected_scope": None,
            "audit_status": "not_attempted",
            "fallback_status": "not_applicable",
        }
        self.traces.append(trace)
        if solved.get("witness_found") is True:
            trace.update(selected_scope="strict", audit_status="pending")
            self._pending = (solved, targets, trace, "strict")
            return solved
        if solved.get("witness_found") is not False:
            raise ValueError("The strict search must explicitly return a candidate or no witness")
        if (
            solved.get("status") != "shared_shift_no_witness"
            or strict_trace.get("primary_status") != "two_step_no_witness"
            or strict_trace.get("fallback_status") != "model_safety_or_first_pose_rejected"
            or strict_trace.get("fallback_rollouts") != 1
            or strict_trace.get("fallback_first_pose_caps_satisfied") is not False
            or strict_trace.get("fallback_safety", {}).get("accepted") is not True
        ):
            trace["fallback_status"] = "strict_stop_not_eligible"
            return solved
        previous_errors = finite(strict_trace.get("fallback_first_two_tcp_errors"), (2, 6))
        if not _bounded_first(previous_errors[0]):
            trace["fallback_status"] = "first_pose_outside_fallback_bound"
            return solved
        if count == MAX_RESPONSES:
            trace["fallback_status"] = "response_budget_exhausted"
            return solved
        shifted = self.primary._completed_shift(seed, previous)
        if shifted is None:
            trace["fallback_status"] = "no_completed_plan"
            return solved
        lower, upper = finite(lower, (7,)), finite(upper, (7,))
        profile = finite(kwargs.get("past_inner_error_rad"), (25, 7))
        callback = kwargs.get("on_response")
        if callback is not None and not callable(callback):
            raise ValueError("A response callback must be callable")
        if callback is not None:
            callback("attempted")
        rollout = rollout_joint_preview(
            model,
            velocity,
            np.asarray([interface.decode(row) for row in shifted]),
            site,
            lower,
            upper,
            enforce_limits=False,
        )
        if callback is not None:
            callback("completed")
        trace.update(extra_rollouts=1, response_evaluations=count + 1, fallback_status="recomputed")
        domain = linear_domain(
            model,
            shifted,
            previous,
            interface,
            lower,
            upper,
            profile,
            rollout,
            terminal_normals=kwargs.get("terminal_normals"),
        )
        safety = domain_check(shifted, domain, interface)
        original_safety = strict_trace["fallback_safety"]
        if any(
            safety.get(key) != original_safety.get(key)
            for key in ("box", "rate", "joint_and_braking", "terminal", "accepted")
        ):
            raise ValueError("Recomputed model safety must match the strict fallback check")
        errors = np.asarray(
            [pose_error(target, pose @ tool) for target, pose in zip(targets, rollout["pose"], strict=True)]
        )
        np.testing.assert_allclose(errors[:2], previous_errors, atol=1e-10, rtol=0)
        if safety["accepted"] is not True or not _bounded_first(errors[0]):
            trace["fallback_status"] = "recomputed_safety_or_bound_rejected"
            return dict(solved, response_evaluations=count + 1)
        witness = {
            "witness_found": True,
            "status": STATUS,
            "selection_scope": STATUS,
            "hard_pose_prefix": 0,
            "response_evaluations": count + 1,
            "planned_commands": shifted,
            "rollout": rollout,
            "first_tcp_error": errors[0].copy(),
            "tcp_errors": errors,
            "terminal_normals": domain["terminal_normals"],
            "quality_slack": _quality_gap(errors),
        }
        trace.update(fallback_status="candidate", selected_scope=STATUS, audit_status="pending")
        self._pending = (witness, targets, trace, "degraded")
        return witness

    def audit(
        self,
        witness,
        mass,
        jacobian,
        kp,
        kd,
        q,
        velocity,
        initial_joint,
        interface,
        previous,
        site,
        site_from_tcp,
        first_target,
        lower,
        upper,
        *,
        past_inner_error_rad,
    ):
        if self._pending is None or witness is not self._pending[0]:
            raise ValueError("The currently selected witness must be audited exactly once")
        _, targets, trace, scope = self._pending
        self._pending = None
        try:
            if scope == "strict":
                verified = self.primary.audit(
                    witness,
                    mass,
                    jacobian,
                    kp,
                    kd,
                    q,
                    velocity,
                    initial_joint,
                    interface,
                    previous,
                    site,
                    site_from_tcp,
                    first_target,
                    lower,
                    upper,
                    past_inner_error_rad=past_inner_error_rad,
                )
            else:
                if not np.array_equal(finite(first_target, (4, 4)), targets[0]):
                    raise ValueError("The first target must match the bound generated reference")
                verified = self.auditor(
                    witness,
                    mass,
                    jacobian,
                    kp,
                    kd,
                    q,
                    velocity,
                    initial_joint,
                    interface,
                    previous,
                    site,
                    site_from_tcp,
                    targets,
                    lower,
                    upper,
                    past_inner_error_rad=past_inner_error_rad,
                    enforce_pose_caps=False,
                    enforce_braking=True,
                )
                if (
                    verified.get("passed") is not True
                    or verified.get("pose_horizon") != 8
                    or verified.get("first_pose_caps_enforced") is not False
                    or verified.get("all_pose_caps_enforced") is not False
                    or verified.get("braking_constraint_enforced") is not True
                    or not _bounded_first(verified["tcp_errors"][0])
                    or not np.isclose(_quality_gap(verified["tcp_errors"]), witness["quality_slack"], atol=1e-8, rtol=0)
                ):
                    raise ValueError("The bounded tracking fallback failed independent model safety or error audit")
                verified.update(selection_scope=STATUS, hard_pose_prefix=0, original_pose_caps_enforced=False)
        except (AssertionError, ValueError, RuntimeError) as error:
            trace.update(audit_status="rejected", audit_error={"type": type(error).__name__, "message": str(error)})
            raise
        trace["audit_status"] = "passed"
        return verified


class ScopedQualityFeedbackRefiner(QualityFeedbackRefiner):
    """Expose a degraded fallback without calling it a pose-quality witness."""

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        if result.get("quality", {}).get("status") == STATUS:
            result["route"] = STATUS
            result["quality"]["original_pose_caps_enforced"] = False
        return result
