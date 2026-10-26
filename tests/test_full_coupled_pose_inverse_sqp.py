import json
from pathlib import Path

import numpy as np
import pytest
import test_full_coupled_pose_inverse as original
from full_coupled_pose_independent import verify_pose_solution

from s2a.full_coupled_pose_inverse_sqp import FullCoupledPoseModel, solve_pose


@pytest.mark.parametrize(
    "behavior",
    [
        original.test_known_full_pose_and_world_twist_recovers_six_commands_with_bias,
        original.test_zero_latch_is_a_distinct_feasible_goal_not_identity,
        original.test_exact_zero_reset_has_explicit_nonzero_native_representative,
        original.test_saturated_solution_has_actual_six_axis_bounds_and_independent_ode_kkt,
        original.test_json_modes_are_independently_verified_and_corruptions_rejected,
        original.test_ambiguous_pi_error_and_infeasible_previous_are_rejected,
    ],
)
def test_sqp_preserves_established_geometry_bounds_and_protocol(monkeypatch, behavior):
    monkeypatch.setattr(original, "solve_pose", solve_pose)
    behavior()


def test_real_false_convergence_fixture_meets_unchanged_independent_stationarity():
    fixture = json.loads((Path(__file__).parent / "fixtures/full_pose_case04_step71.json").read_text())
    contract = fixture["contract"]
    model = FullCoupledPoseModel(fixture["mass"], fixture["jacobian"], contract["kp"], contract["kd"])
    result = solve_pose(
        model,
        fixture["measured"],
        fixture["desired"],
        fixture["twist"],
        fixture["reference_twist"],
        fixture["previous"],
        fixture["previous_goal"],
        contract,
        innovation=fixture["innovation"],
    )
    wire = json.loads(json.dumps(result, default=lambda value: value.tolist(), allow_nan=False))
    certificate = verify_pose_solution(
        wire,
        fixture["mass"],
        fixture["jacobian"],
        contract,
        fixture["measured"],
        fixture["desired"],
        fixture["twist"],
        fixture["reference_twist"],
        fixture["previous"],
        fixture["previous_goal"],
        fixture["innovation"],
    )
    assert certificate["max_independent_stationarity"] < 2e-6
    assert all(candidate["solver_method"] == "SLSQP" for candidate in wire["mode_candidates"])
    assert wire["mode_objectives"]["reset"] < 0.02189192179345653
    assert np.max(np.abs(np.asarray(wire["command"]) - fixture["previous"])) <= 0.25 + 1e-12
