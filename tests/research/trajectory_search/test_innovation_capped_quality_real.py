"""Exercise the actual isolated optimizer with an impossible shifted pose gate."""

import numpy as np

from s2a.research.trajectory_search import innovation_capped_quality_solver as solver
from tests.research.joint_control.terminal.test_configurable_tracking_terminal_plan import inputs


def test_actual_solver_keeps_original_namespace_and_rejects_far_innovation():
    original_minimize = solver.quality.minimize
    result = solver.solve_interior_plan(
        *inputs(),
        translation_innovation=np.array([1.0, 0, 0]),
        rotation_discrepancy=np.eye(3),
        past_inner_error_rad=np.zeros((25, 7)),
    )
    assert solver.quality.minimize is original_minimize
    assert result["witness_found"] is False
    assert "planned_commands" not in result
    assert 0 < result["response_evaluations"] <= 160
