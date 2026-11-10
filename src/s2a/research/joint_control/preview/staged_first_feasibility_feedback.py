"""Record staged first-pose feasibility within the existing quality audit."""

import copy

from s2a.research.joint_control.preview.feasible_start_restoration_feedback import FeasibleStartRestorationSolverAudit
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.staged_first_feasibility import solve_staged_first_feasibility


class StagedFirstFeasibilitySolverAudit(FeasibleStartRestorationSolverAudit):
    """Account for both stages and retain the existing independent audit."""

    def __init__(self, receipt, *, solver=solve_staged_first_feasibility, auditor=audit_full_horizon_witness):
        if not isinstance(receipt, dict) or "staged_first_feasibility_traces" in receipt:
            raise ValueError("One fresh staged-search receipt is required")
        self.stage_traces = receipt["staged_first_feasibility_traces"] = []

        def recorded_solver(*args, **kwargs):
            solved = solver(*args, **kwargs)
            phases = solved.get("search_phases")
            count = solved.get("response_evaluations")
            if not isinstance(phases, list) or not isinstance(count, int) or not 0 <= count <= 320:
                raise ValueError("Staged search must account for at most 320 responses")
            names = [phase.get("phase") for phase in phases]
            if names not in ([], ["first_feasibility"], ["first_feasibility", "future_quality"]):
                raise ValueError("Staged search phase ordering is invalid")
            if phases:
                first = phases[0].get("iterations")
                if first is not None and (not isinstance(first, int) or not 0 <= first <= 40):
                    raise ValueError("First feasibility exceeds its iteration budget")
                if len(phases) == 2:
                    second = phases[1].get("iterations")
                    if first is None or (
                        second is not None and (not isinstance(second, int) or not 0 <= second <= 120 - first)
                    ):
                        raise ValueError("Staged search exceeds its shared iteration budget")
            self.stage_traces.append(
                dict(status=solved.get("status"), response_evaluations=count, phases=copy.deepcopy(phases))
            )
            return solved

        super().__init__(receipt, solver=recorded_solver, auditor=auditor)
