"""Bind each eight-point feedback solve to its own independent reference audit."""

import numpy as np

from s2a.geometry import finite
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.full_horizon_quality import solve_full_horizon


class FullHorizonSolverAudit:
    """Adapt eight-point solver/audit to the existing quality transaction."""

    def __init__(self, *, solver=solve_full_horizon, auditor=audit_full_horizon_witness):
        self._solver, self._auditor = solver, auditor
        self._pending = None

    def solve(self, model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper, **kwargs):
        if self._pending is not None:
            raise ValueError("The previous quality witness must be audited before another solve")
        targets = finite(targets, (8, 4, 4)).copy()
        if finite(seed).shape != (8, 6) or finite(twists).shape != (8, 6):
            raise ValueError("Exactly eight current generated targets and commands are required")
        solver_targets = targets.copy()
        result = self._solver(
            model, velocity, seed, previous, interface, site, tool, solver_targets, twists, lower, upper, **kwargs
        )
        if not np.array_equal(solver_targets, targets):
            raise ValueError("The solver must not mutate its bound reference")
        if result.get("witness_found") is True:
            self._pending = (result, targets)
        return result

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
        )
        if (
            verified.get("passed") is not True
            or verified.get("pose_horizon") != 8
            or verified.get("all_pose_caps_satisfied") is not True
            or verified.get("all_pose_caps_enforced") is not True
        ):
            raise ValueError("All eight predicted poses must pass the independent quality audit")
        return verified
