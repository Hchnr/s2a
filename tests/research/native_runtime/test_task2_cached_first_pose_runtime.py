"""Cached first-pose candidates require an independent complete safety audit."""

import numpy as np

from s2a.research.trajectory_search.cached_first_pose import collect_cached_first_pose
from scripts.tracking.trajectory_search import audit_task2_cached_first_pose as runner
from scripts.tracking.trajectory_search import compare_task2_shifted_seed_search as comparison
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_comparison_instrumentation_does_not_change_original_globals():
    captures = []
    candidate = runner.isolated_analyze(captures)
    assert candidate.__code__ is comparison.analyze.__code__
    assert candidate.__globals__ is not comparison.analyze.__globals__
    assert candidate.__globals__["solve_guarded_two_step"] is not comparison.solve_guarded_two_step


def test_cached_first_candidate_gets_independent_eight_period_audit():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 1.0
    profile = np.zeros((25, 7))
    result = collect_cached_first_pose(*args, past_inner_error_rad=profile)
    assert result["primary"]["status"] == "two_step_no_witness" and result["best_first_pose"] is not None
    mass, jacobian, kp, kd, q = audit_args[:5]
    case = dict(controller=dict(contract=dict(kp=kp, kd=kd)), initial_joint=q)
    np.testing.assert_array_equal(args[0]["mass"], mass)
    np.testing.assert_array_equal(args[0]["jacobian"], jacobian)
    stats = comparison.new_stats()
    checked = runner.audit_cached((args, {"past_inner_error_rad": profile}, result), case, stats)
    assert checked["status"] == "passed" and checked["verification"]["first_pose_caps_satisfied"]
    assert stats["independent_audits_attempted"] == stats["independent_audits_passed"] == 1
    assert stats["independent_endpoint_queries_charged"] == 200
    assert stats["independent_transition_steps_charged"] == 2600
