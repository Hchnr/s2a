"""Opt-in reuse of an already evaluated nominal two-pose witness."""

import copy
from types import FunctionType, SimpleNamespace

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.preview.shared_shift_fallback import SharedShiftFallbackAudit
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search import margin_guarded_fallback as shared


def _clone(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


def solve_with_nominal_capture(capture, *args, **kwargs):
    """Run the unchanged shared solver while retaining its primary nominal candidate."""
    if not isinstance(capture, dict) or capture:
        raise ValueError("A fresh nominal-candidate capture is required")
    isolated = shared.solve_margin_guarded_fallback
    namespace = isolated.__globals__.copy()
    original_margin = namespace["isolated_margin_solver"]

    def isolated_margin(domain_capture):
        margin_solver = original_margin(domain_capture)
        margin_namespace = margin_solver.__globals__.copy()
        guarded = margin_namespace["guarded"]

        def isolated_guarded():
            solver = guarded.isolated_solver()
            deepcopy = copy.deepcopy

            def observed(*solver_args, **solver_kwargs):
                result = solver(*solver_args, **solver_kwargs)
                if capture:
                    raise ValueError("Exactly one primary nominal result may be observed")
                candidate = result.get("best_joint_safe")
                capture.update(
                    primary_status=result.get("status"),
                    primary_response_evaluations=result.get("response_evaluations"),
                    nominal_prefix_evaluations=result.get("prefix_hard_evaluations"),
                    candidate=None if candidate is None else deepcopy(candidate),
                )
                return result

            # The margin wrapper instruments the returned callable's globals;
            # retain the solver's exact isolated namespace for those hooks.
            return _clone(observed, solver.__globals__)

        margin_namespace["guarded"] = SimpleNamespace(
            isolated_solver=isolated_guarded, OBJECTIVE_SCALE=guarded.OBJECTIVE_SCALE
        )
        return _clone(margin_solver, margin_namespace)

    namespace["isolated_margin_solver"] = isolated_margin
    return _clone(isolated, namespace)(*args, **kwargs)


class CachedNominalFallbackAudit(SharedShiftFallbackAudit):
    """Try the original complete fallback first, then audit a captured nominal plan."""

    def __init__(self, receipt):
        if not isinstance(receipt, dict) or "cached_nominal_traces" in receipt:
            raise ValueError("One fresh cached-nominal receipt is required")
        self.enabled = False
        self._captured_nominal = {}
        self.nominal_traces = receipt["cached_nominal_traces"] = []

        def solver(*args, **kwargs):
            if not self.enabled:
                return shared.solve_margin_guarded_fallback(*args, **kwargs)
            return solve_with_nominal_capture(self._captured_nominal, *args, **kwargs)

        super().__init__(receipt, solver=solver)
        original_dual = self._dual_auditor

        def dual_auditor(witness, *args, **kwargs):
            if witness.get("selection_scope") != "cached_nominal_two_step":
                return original_dual(witness, *args, **kwargs)
            verified = audit_two_step_witness(witness, *args, **kwargs)
            verified.update(selection_scope="cached_nominal_two_step", model_margin_enforced=False)
            return verified

        self._dual_auditor = dual_auditor

    def activate(self, policy, owner, receipt, *, step):
        """Enable once at a completed, idle, causally measured boundary."""
        arm, inner = policy.arm_policy, policy.inner_history
        tracking = arm.tracking
        past, gripper = inner.snapshot(), policy.gripper.state()
        if (
            self.enabled
            or type(step) is not int
            or step <= 0
            or type(receipt) is not dict
            or "cached_nominal_activation" in receipt
            or receipt.get("control_steps_completed") != step
            or not 0 < step - policy.start_step < 8
            or policy.max_steps != 8
            or policy.pending is not None
            or policy.last_inner_loss
            or getattr(inner, "_owner", None) is not owner
            or past.get("completed_periods") != step - policy.start_step
            or past.get("last_completed_step") != step - 1
            or past.get("future_guarantee") is not False
            or arm.pending is not None
            or arm.coupled.pending is not None
            or arm.coupled.completed_periods != step
            or tracking.phase != "ready"
            or tracking.history is not self._history
            or tracking.history.completed_periods != step
            or tracking.history.pending is not None
            or tracking.history.pending_plan is not None
            or tracking._refiner.history is not inner
            or getattr(tracking._refiner.solver, "__self__", None) is not self
            or self._pending is not None
            or gripper.get("staged") is not False
            or gripper.get("faulted") is not False
        ):
            raise ValueError("The exact completed and idle feedback boundary is required")
        receipt["cached_nominal_activation"] = dict(step=step, completed_periods=step, future_observations=0)
        self.enabled = True

    def solve(self, model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs):
        self._captured_nominal = {}
        result = super().solve(
            model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs
        )
        if not self.enabled:
            return result
        captured = self._captured_nominal
        candidate = captured.get("candidate")
        trace = dict(
            enabled=True,
            primary_status=captured.get("primary_status"),
            primary_response_evaluations=captured.get("primary_response_evaluations"),
            nominal_prefix_evaluations=captured.get("nominal_prefix_evaluations"),
            baseline_status=result.get("status"),
            baseline_witness_found=result.get("witness_found"),
            candidate_available=candidate is not None,
            selected=False,
            extra_model_responses=0,
        )
        self.nominal_traces.append(trace)
        if result.get("witness_found") is True or candidate is None:
            return result
        count = result.get("response_evaluations")
        errors = finite(candidate.get("tcp_errors"), (8, 6))
        if (
            captured.get("primary_status") != "two_step_candidate"
            or type(captured.get("nominal_prefix_evaluations")) is not int
            or captured["nominal_prefix_evaluations"] < 1
            or type(count) is not int
            or type(captured.get("primary_response_evaluations")) is not int
            or not 0 <= captured["primary_response_evaluations"] <= count <= 320
            or candidate.get("witness_found") is not True
            or candidate.get("hard_pose_prefix") != 2
            or candidate.get("prefix_selection_hard") is not True
            or np.any(np.linalg.norm(errors[:2, :3], axis=1) > 0.01 + 1e-10)
            or np.any(np.linalg.norm(errors[:2, 3:], axis=1) > 0.05 + 1e-10)
            or self._pending is not None
        ):
            raise ValueError("Only an already evaluated nominal two-pose candidate may be reused")
        finite(candidate.get("planned_commands"), (8, 6))
        selected = dict(
            candidate,
            status="cached_nominal_two_step_candidate",
            selection_scope="cached_nominal_two_step",
            response_evaluations=count,
        )
        causal, shared_trace = self.traces[-1], self.shared_traces[-1]
        causal.update(selected_scope="two_step", audit_status="pending", cached_nominal_reused=True)
        shared_trace.update(
            selected_scope="cached_nominal_two_step", audit_status="pending", cached_nominal_reused=True
        )
        self._pending = (selected, finite(targets, (8, 4, 4)).copy(), causal)
        trace["selected"] = True
        return selected
