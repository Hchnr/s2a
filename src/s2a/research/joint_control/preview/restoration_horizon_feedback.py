"""Restore tracking progress with independently checked first-point quality."""

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.preview.full_horizon_feedback import FullHorizonSolverAudit
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.full_horizon_quality import solve_full_horizon


class RestorationHorizonSolverAudit(FullHorizonSolverAudit):
    """Keep joint constraints hard while exposing future pose-quality shortfall.

    The transaction witness is a joint-model candidate, not an assertion that
    all eight pose caps pass. Independent audit must also pass the first pose.
    """

    def __init__(self, *, solver=solve_full_horizon, auditor=audit_full_horizon_witness):
        self._restoration_solver = solver
        super().__init__(solver=self._restore, auditor=auditor)

    def _restore(self, *args, **kwargs):
        if "feasibility_restoration" in kwargs or "enforce_braking" in kwargs:
            raise ValueError("Restoration and braking are fixed by this candidate")
        solved = self._restoration_solver(*args, feasibility_restoration=True, **kwargs)
        candidate = solved.get("best_joint_safe")
        if candidate is None:
            return {
                "witness_found": False,
                "status": str(solved["status"]),
                "response_evaluations": solved["response_evaluations"],
            }
        candidate = dict(candidate)
        if candidate.get("witness_found") is not True:
            raise ValueError("Restoration must return a joint-model candidate")
        candidate.update(
            status="restoration_joint_candidate",
            response_evaluations=solved["response_evaluations"],
            restoration_optimizer_status=str(solved["status"]),
        )
        return candidate

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
            raise ValueError("The auditor requires the current solver witness")
        targets = self._pending[1]
        self._pending = None
        if not np.array_equal(finite(first_target, (4, 4)), targets[0]):
            raise ValueError("The first target must match the bound current reference")
        verified = self._auditor(
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
        )
        if (
            verified.get("passed") is not True
            or verified.get("pose_horizon") != 8
            or verified.get("braking_constraint_enforced") is not True
            or verified.get("first_pose_caps_satisfied") is not True
            or verified.get("all_pose_caps_enforced") is not False
        ):
            raise ValueError("The first pose and all hard model constraints must pass independently")
        errors = finite(verified["tcp_errors"], (8, 6))
        slack = max(
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
        if not np.isclose(slack, witness["quality_slack"], atol=1e-8, rtol=0):
            raise ValueError("The independent eight-point quality gap must match the candidate")
        verified.update(first_pose_caps_enforced=True, quality_slack=slack, selection_scope="joint_and_first_pose")
        return verified
