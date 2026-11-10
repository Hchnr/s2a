"""Preserve feasible-start costs within the existing first-hard audit."""

import copy

from s2a.research.joint_control.preview.first_hard_restoration_feedback import FirstHardRestorationSolverAudit
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.feasible_start_restoration import solve_feasible_start_restoration


class FeasibleStartRestorationSolverAudit(FirstHardRestorationSolverAudit):
    """Record initializer LPs and delegate all original dispatch audits."""

    def __init__(self, receipt, *, solver=solve_feasible_start_restoration, auditor=audit_full_horizon_witness):
        if not isinstance(receipt, dict) or "feasible_start_traces" in receipt:
            raise ValueError("One fresh initializer receipt is required")
        self.start_traces = receipt["feasible_start_traces"] = []

        def recorded_solver(*args, **kwargs):
            solved = solver(*args, **kwargs)
            info = solved.get("initialization")
            if not isinstance(info, dict) or not 0 <= info.get("start_lp_attempts", -1) <= 2:
                raise ValueError("Every initialized solve must account for at most two LPs")
            if info.get("actual_model_rollouts") != solved.get("response_evaluations"):
                raise ValueError("Initializer and solver model response counts must match")
            self.start_traces.append(copy.deepcopy(info))
            return solved

        super().__init__(receipt, solver=recorded_solver, auditor=auditor)
