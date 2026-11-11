"""Automatic solver evaluation uses current causal inputs and independent audit."""

import numpy as np

from scripts.tracking.trajectory_search import run_cached_live_stop125_feasibility as base
from scripts.tracking.trajectory_search import run_quality_rate_repair_solver as run


def test_solver_uses_current_seed_and_external_audit_controls_qualification(monkeypatch):
    seed = np.full((8, 6), 0.1)
    stopped = {"previous": np.zeros(7), "shifted_seed": seed, "past_inner_error_rad": np.zeros((25, 7))}
    targets = np.tile(np.eye(4), (40, 1, 1))
    context = {
        "model": object(),
        "velocity": np.zeros(7),
        "interface": object(),
        "site": np.eye(4),
        "site_from_tcp": np.eye(4),
        "limits": np.tile([-2.0, 2.0], (7, 1)),
        "mass": np.eye(7),
        "jacobian": np.zeros((6, 7)),
        "q": np.zeros(7),
        "contract": {"kp": np.ones(6), "kd": np.ones(6)},
        "initial_joint": np.zeros(7),
    }

    for audit_passes in (False, True):
        stats = base.new_stats()

        def fake_solver(*args, **kwargs):
            np.testing.assert_array_equal(args[2], seed)
            np.testing.assert_array_equal(args[7], targets[5:13])
            assert set(kwargs) == {"past_inner_error_rad"}
            stats["optimizer_responses_attempted"] += 1
            stats["optimizer_responses_completed"] += 1
            return {"witness_found": True, "response_evaluations": 1, "planned_commands": seed, "rollout": {}}

        def auditor(*_args, **kwargs):
            assert kwargs["enforce_pose_caps"] is True
            if not audit_passes:
                raise ValueError("independent rejection")
            return {
                "passed": True,
                "first_pose_caps_satisfied": True,
                "first_pose_caps_enforced": True,
                "independent_endpoint_queries": 200,
                "independent_transition_steps": 2600,
            }

        monkeypatch.setattr(run.solver, "solve_interior_plan", fake_solver)
        result = run.single_search(
            context,
            stopped,
            targets,
            np.eye(4),
            stats,
            quality_solver=None,
            joint_solver=None,
            auditor=auditor,
            results={},
        )
        assert result["quality_rate_repair"]["certified_model_witness"] is audit_passes
        assert stats["optimizer_responses_completed"] == 1
        assert stats["independent_endpoint_queries_charged"] == 200
