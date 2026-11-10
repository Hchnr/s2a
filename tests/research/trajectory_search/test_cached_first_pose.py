"""Cache inspection must preserve the primary search and its response budget."""

import numpy as np
import pytest

from s2a.infrastructure.publication import result_json
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search import guarded_two_step_search as guarded
from s2a.research.trajectory_search.cached_first_pose import collect_cached_first_pose
from tests.research.joint_control.preview.test_full_horizon_feedback import solve_and_audit_inputs


def test_cache_instrumentation_preserves_primary_solver_and_response_count():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 0.015
    targets[-1, 0, 3] += 0.2
    before = guarded.isolated_solver().__globals__["domain_check"]
    primary = guarded.solve_guarded_two_step(*args)
    inspected = collect_cached_first_pose(*args)
    assert result_json(inspected["primary"]) == result_json(primary)
    assert inspected["recorded_rollout_calls"] == primary["response_evaluations"] <= 320
    assert inspected["unique_safe_candidates"] == primary["joint_safe_evaluations"]
    assert inspected["extra_model_responses"] == 0
    assert guarded.isolated_solver().__globals__["domain_check"] is before


def test_cache_can_find_an_independently_safe_first_pose_when_second_is_unreachable():
    args, audit_args, targets = solve_and_audit_inputs()
    targets[1, 0, 3] += 1.0
    inspected = collect_cached_first_pose(*args)
    assert inspected["primary"]["status"] == "two_step_no_witness"
    witness = inspected["best_first_pose"]
    assert witness is not None and inspected["first_pose_candidates"] > 0
    audit_args = list(audit_args)
    audit_args[11] = targets
    checked = audit_full_horizon_witness(
        witness, *audit_args, past_inner_error_rad=np.zeros((25, 7)), enforce_pose_caps=False, enforce_braking=True
    )
    assert checked["passed"] and checked["first_pose_caps_satisfied"] and checked["braking_constraint_satisfied"]
    assert np.linalg.norm(checked["tcp_errors"][1, :3]) > 0.01


def test_observed_cache_records_no_candidate_when_first_target_is_unreachable():
    args, _audit_args, targets = solve_and_audit_inputs()
    targets[0, 0, 3] += 3.0
    inspected = collect_cached_first_pose(*args)
    assert inspected["primary"]["status"] == "two_step_no_witness"
    assert inspected["first_pose_candidates"] == 0 and inspected["best_first_pose"] is None
    assert inspected["recorded_rollout_calls"] <= 320


def test_invalid_target_horizon_is_rejected_before_search():
    args, _audit_args, _targets = solve_and_audit_inputs()
    changed = list(args)
    changed[7] = changed[7][:7]
    with pytest.raises(ValueError):
        collect_cached_first_pose(*changed)
