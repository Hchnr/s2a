import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation

from s2a.coupled_pose_preview import plan_pose
from s2a.current_coupled_response import build_response


def problem():
    f = json.loads((Path(__file__).parent / "fixtures/full_pose_case04_step71.json").read_text())
    return dict(
        measured=np.asarray(f["measured"]),
        twist=np.asarray(f["twist"]),
        references=np.repeat(np.asarray(f["desired"])[None], 4, axis=0),
        reference_twists=np.repeat(np.asarray(f["reference_twist"])[None], 4, axis=0),
        previous_command=np.asarray(f["previous"]),
        mass=f["mass"],
        jacobian=f["jacobian"],
        contract=f["contract"],
        innovation=np.asarray(f["innovation"]),
    )


def test_real_fixture_explicit_propagation_and_independent_finite_difference_kkt():
    p = problem()
    result = plan_pose(**p)
    c = p["contract"]
    model = build_response(p["mass"], p["jacobian"], c["kp"], c["kd"], np.eye(3))
    targets = np.c_[
        p["references"][:, :3, 3] - p["measured"][:3, 3],
        Rotation.from_matrix(p["references"][:, :3, :3] @ p["measured"][:3, :3].T).as_rotvec(),
        p["reference_twists"],
    ]
    weight = np.array([100] * 3 + [20] * 3 + [5] * 3 + [1] * 3)

    def evaluate(flat):
        position, velocity = np.zeros(6), p["twist"].copy()
        states = []
        for u in flat.reshape(4, 6):
            delta = (u - (np.asarray(c["input_max"]) + c["input_min"]) / 2) * (
                (np.asarray(c["output_max"]) - c["output_min"]) / (np.asarray(c["input_max"]) - c["input_min"])
            )
            delta += (np.asarray(c["output_max"]) + c["output_min"]) / 2
            motion = model["initial_velocity_response"] @ velocity + model["goal_forcing_response"] @ delta
            position += motion[:6] + p["innovation"][:6]
            velocity = motion[6:] + p["innovation"][6:] / 0.05
            states.append(np.r_[position, velocity])
        error = (np.asarray(states) - targets) * weight
        return 0.5 * np.sum(error**2), np.asarray(states)

    x = result["planned_commands"].ravel()
    value, states = evaluate(x)
    np.testing.assert_allclose(states, result["predicted_tangent_states"], atol=1e-12)
    np.testing.assert_allclose(value, result["objective"], atol=1e-10)
    gradient = np.array(
        [(evaluate(x + np.eye(24)[i] * 1e-5)[0] - evaluate(x - np.eye(24)[i] * 1e-5)[0]) / 2e-5 for i in range(24)]
    )
    identity = np.eye(24)
    difference = identity - np.eye(24, k=-6)
    constraints = np.vstack([identity, -identity, difference, -difference])
    previous = np.r_[p["previous_command"], np.zeros(18)]
    bounds = np.r_[np.tile(c["input_max"], 4), -np.tile(c["input_min"], 4), 0.25 + previous, 0.25 - previous]
    slack = bounds - constraints @ x
    assert slack.min() >= -1e-9
    active = constraints[slack < 1e-8]
    multipliers = nnls(active.T, -gradient, maxiter=10000)[0]
    assert np.max(np.abs(gradient + active.T @ multipliers)) < 2e-6


def test_future_generated_reference_changes_first_command_without_mutating_inputs():
    p = problem()
    original = p["references"].copy()
    first = plan_pose(**p)
    np.testing.assert_array_equal(p["references"], original)
    p["references"][1:, 0, 3] += 0.04
    second = plan_pose(**p)
    assert np.linalg.norm(first["command"] - second["command"]) > 1e-4
    assert first["native_decode_and_actual_commit_required"] and not first["nonlinear_optimality_claim"]


@pytest.mark.parametrize("failure", ["previous", "pi", "horizon", "nonfinite"])
def test_invalid_or_ambiguous_online_inputs_rejected(failure):
    p = problem()
    if failure == "previous":
        p["previous_command"][0] = 2
    if failure == "pi":
        p["references"][0, :3, :3] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix() @ p["measured"][:3, :3]
    if failure == "horizon":
        p["references"] = np.repeat(p["references"][:1], 5, axis=0)
    if failure == "nonfinite":
        p["innovation"][0] = np.nan
    with pytest.raises(ValueError):
        plan_pose(**p)
