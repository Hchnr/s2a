"""Prefer two audited poses, with a past-plan fallback for the first pose."""

import copy

import numpy as np

from s2a.geometry import finite, pose_error
from s2a.research.joint_control.preview.interior_joint_preview import rollout_joint_preview
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.joint_control.tracking.two_step_quality_audit import audit_two_step_witness
from s2a.research.trajectory_search.adaptive_plan_history import AdaptivePlanHistory
from s2a.research.trajectory_search.feasible_start_restoration import domain_check, linear_domain
from s2a.research.trajectory_search.guarded_two_step_search import (
    OBJECTIVE_SCALE,
    SEARCH_GUARD,
    solve_guarded_two_step,
)


class CausalShiftFallbackAudit:
    """Audit the primary pair, or a completed shifted plan's first pose.

    A fallback rollout counts against the same 320-response limit. History is
    attached after construction so the original feedback transaction owns it.
    """

    def __init__(
        self,
        receipt,
        *,
        solver=solve_guarded_two_step,
        dual_auditor=audit_two_step_witness,
        first_auditor=audit_full_horizon_witness,
    ):
        if not isinstance(receipt, dict) or "causal_shift_fallback_traces" in receipt:
            raise ValueError("One fresh fallback receipt is required")
        self.traces = receipt["causal_shift_fallback_traces"] = []
        self._solver, self._dual_auditor, self._first_auditor = solver, dual_auditor, first_auditor
        self._history = None
        self._pending = None

    def attach_history(self, history):
        if self._history is not None or not isinstance(history, AdaptivePlanHistory):
            raise ValueError("Exactly one branch-local completed-plan history is required")
        self._history = history

    def _completed_shift(self, seed, previous):
        history = self._history
        if history is None:
            raise ValueError("The fallback requires the transaction's history")
        if history.pending_plan is None or not np.array_equal(history.pending_plan, seed):
            raise ValueError("The baseline must stage the currently bound seed")
        if history.last_plan is None:
            return None
        if history.completed_periods < 1 or history.last_plan.shape != (8, 6):
            raise ValueError("Only a completed eight-period plan may be shifted")
        if not np.array_equal(history.last_plan[0], previous):
            raise ValueError("The completed plan must match the last dispatched command")
        return finite(history.shifted(8), (8, 6)).copy()

    @staticmethod
    def _validate_primary(solved):
        details = solved.get("first_stage_search_objective")
        if details != dict(
            normalized_guard=SEARCH_GUARD,
            scale=OBJECTIVE_SCALE,
            acceptance_caps_changed=False,
            future_stage_changed=False,
        ):
            raise ValueError("The original guarded objective and acceptance must remain fixed")
        count, info, phases = (
            solved.get("response_evaluations"),
            solved.get("initialization"),
            solved.get("search_phases"),
        )
        if (
            type(count) is not int
            or not 0 <= count <= 320
            or not isinstance(info, dict)
            or type(info.get("start_lp_attempts")) is not int
            or not 0 <= info["start_lp_attempts"] <= 2
            or not isinstance(phases, list)
        ):
            raise ValueError("The primary search must account for its original bounded costs")
        names = [phase.get("phase") for phase in phases]
        if names not in ([], ["two_point_feasibility"], ["two_point_feasibility", "future_quality"]):
            raise ValueError("The original two-point phase ordering is required")
        if phases:
            first = phases[0].get("iterations")
            if first is not None and (type(first) is not int or not 0 <= first <= 60):
                raise ValueError("The original first-stage iteration budget is required")
            if len(phases) == 2:
                second = phases[1].get("iterations")
                if first is None or (
                    second is not None and (type(second) is not int or not 0 <= second <= 120 - first)
                ):
                    raise ValueError("The original shared iteration budget is required")
        return count

    def solve(self, model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs):
        if self._pending is not None:
            raise ValueError("The previous selected plan must be audited first")
        seed, previous, targets = finite(seed, (8, 6)), finite(previous, (6,)), finite(targets, (8, 4, 4)).copy()
        if finite(twists).shape != (8, 6):
            raise ValueError("Eight generated target twists are required")
        solver_targets = targets.copy()
        solved = self._solver(
            model, velocity, seed, previous, interface, site, tool, solver_targets, twists, lower, upper, **kwargs
        )
        if not np.array_equal(solver_targets, targets):
            raise ValueError("The primary solver cannot change generated targets")
        count = self._validate_primary(solved)
        candidate = solved.get("best_joint_safe")
        trace = dict(
            primary_status=solved.get("status"),
            primary_response_evaluations=count,
            primary_initialization=copy.deepcopy(solved["initialization"]),
            primary_phases=copy.deepcopy(solved["search_phases"]),
            fallback_status="not_applicable",
            fallback_rollouts=0,
            selected_scope=None,
            audit_status="not_attempted",
            response_evaluations=count,
        )
        self.traces.append(trace)
        if candidate is not None:
            if (
                solved.get("status") != "two_step_candidate"
                or candidate.get("witness_found") is not True
                or candidate.get("hard_pose_prefix") != 2
                or candidate.get("prefix_selection_hard") is not True
            ):
                raise ValueError("The primary candidate must retain both hard poses")
            selected = dict(candidate, status="guarded_two_step_candidate", response_evaluations=count)
            trace.update(selected_scope="two_step", audit_status="pending")
            self._pending = (selected, targets, trace)
            return selected
        if solved.get("status") != "two_step_no_witness":
            return dict(witness_found=False, status=str(solved.get("status")), response_evaluations=count)
        if count >= 320:
            trace["fallback_status"] = "response_budget_exhausted"
            return dict(witness_found=False, status="two_step_no_witness", response_evaluations=count)
        shifted = self._completed_shift(seed, previous)
        if shifted is None:
            trace["fallback_status"] = "no_completed_plan"
            return dict(witness_found=False, status="two_step_no_witness", response_evaluations=count)
        lower, upper = finite(lower, (7,)), finite(upper, (7,))
        if (
            np.any(shifted < interface.input_min - 1e-12)
            or np.any(shifted > interface.input_max + 1e-12)
            or np.max(abs(np.diff(np.vstack((previous, shifted)), axis=0))) > 0.25 + 1e-12
            or np.any(np.all(np.asarray([interface.decode(row) for row in shifted])[:, 3:] == 0, axis=1))
        ):
            trace["fallback_status"] = "command_or_latch_rejected"
            return dict(witness_found=False, status="two_step_no_witness", response_evaluations=count)
        if np.min(np.minimum(model["q"] - lower, upper - model["q"])) < 0.05 - 1e-12:
            trace["fallback_status"] = "initial_joint_buffer_rejected"
            return dict(witness_found=False, status="two_step_no_witness", response_evaluations=count)
        profile = finite(kwargs.get("past_inner_error_rad"), (25, 7))
        trace.update(fallback_status="rollout_attempted", fallback_rollouts=1, response_evaluations=count + 1)
        rollout = rollout_joint_preview(
            model,
            velocity,
            np.asarray([interface.decode(row) for row in shifted]),
            site,
            lower,
            upper,
            enforce_limits=False,
        )
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
        errors = np.asarray(
            [pose_error(target, pose @ tool) for target, pose in zip(targets, rollout["pose"], strict=True)]
        )
        pos, rot = np.linalg.norm(errors[:, :3], axis=1), np.linalg.norm(errors[:, 3:], axis=1)
        first_ok = bool(pos[0] <= 0.01 + 1e-10 and rot[0] <= 0.05 + 1e-10)
        trace.update(
            fallback_safety=safety,
            fallback_first_two_tcp_errors=errors[:2].copy(),
            fallback_first_pose_caps_satisfied=first_ok,
        )
        if not safety["accepted"] or not first_ok:
            trace["fallback_status"] = "model_safety_or_first_pose_rejected"
            return dict(witness_found=False, status="two_step_no_witness", response_evaluations=count + 1)
        gap = max(0.0, float(np.max(np.r_[pos / 0.01 - 1, rot / 0.05 - 1])))
        selected = dict(
            witness_found=True,
            status="shifted_first_pose_candidate",
            response_evaluations=count + 1,
            planned_commands=shifted,
            rollout=rollout,
            first_tcp_error=errors[0].copy(),
            tcp_errors=errors,
            terminal_normals=domain["terminal_normals"],
            quality_slack=gap,
        )
        trace.update(fallback_status="first_pose_candidate", selected_scope="first_only", audit_status="pending")
        self._pending = (selected, targets, trace)
        return selected

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
            raise ValueError("The auditor requires the current selected plan")
        _selected, targets, trace = self._pending
        self._pending = None
        if not np.array_equal(finite(first_target, (4, 4)), targets[0]):
            trace["audit_status"] = "rejected"
            raise ValueError("The first target must match the bound generated reference")
        args = (
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
        )
        try:
            if trace["selected_scope"] == "two_step":
                verified = self._dual_auditor(*args, past_inner_error_rad=past_inner_error_rad)
                if verified.get("passed") is not True or verified.get("prefix_pose_caps_satisfied") is not True:
                    raise ValueError("The primary two-point audit must pass")
            else:
                verified = self._first_auditor(
                    *args, past_inner_error_rad=past_inner_error_rad, enforce_pose_caps=False, enforce_braking=True
                )
                if (
                    verified.get("passed") is not True
                    or verified.get("pose_horizon") != 8
                    or verified.get("first_pose_caps_satisfied") is not True
                    or verified.get("braking_constraint_enforced") is not True
                    or verified.get("all_pose_caps_enforced") is not False
                ):
                    raise ValueError("The fallback's first pose and full safety audit must pass")
                errors = finite(verified["tcp_errors"], (8, 6))
                gap = max(
                    0.0,
                    float(
                        np.max(
                            np.r_[
                                np.linalg.norm(errors[:, :3], axis=1) / 0.01 - 1,
                                np.linalg.norm(errors[:, 3:], axis=1) / 0.05 - 1,
                            ]
                        )
                    ),
                )
                if not np.isclose(gap, witness["quality_slack"], atol=1e-8, rtol=0):
                    raise ValueError("The independently reconstructed future gap must match")
                verified.update(
                    first_pose_caps_enforced=True,
                    selection_scope="shifted_plan_first_pose",
                    hard_pose_prefix=1,
                    quality_slack=gap,
                )
        except (AssertionError, ValueError, RuntimeError) as error:
            trace.update(audit_status="rejected", audit_error=dict(type=type(error).__name__, message=str(error)))
            raise
        trace["audit_status"] = "passed"
        return verified
