"""Geometry, strict floating zero-set and causal GN contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from s2a.research.trajectory_search.cap_distance_objective import (
    CapDistanceObjective,
    cap_loss,
    linearize,
)
from s2a.research.trajectory_search.state_vector_objective_scale import build_stage_problem, metrics
from tests.research.trajectory_search.test_state_vector_gn_search import entry, inputs


def record(vector):
    E = np.zeros(72)
    E[:3] = vector
    return entry(np.zeros((12, 6)), E)


@pytest.mark.parametrize("vector", [[0, 0, 0], [0.5, 0.2, 0], [1, 0, 0]])
def test_closed_ball_and_boundary_choose_zero_projection(vector):
    model = linearize(record(vector), np.eye(72))
    assert model["loss"] == 0
    assert_array_equal(model["residual"], np.zeros(72))
    assert_array_equal(model["jacobian"], np.zeros((72, 72)))


def test_full_radial_and_tangential_derivatives():
    model = linearize(record([2, 0, 0]), np.eye(72))
    assert_array_equal(model["residual"][:3], [1, 0, 0])
    assert_array_equal(model["projection_blocks"][0], np.diag([1, 0.5, 0.5]))
    assert model["loss"] == 0.5
    vector = np.array([1.4, 0.7, 0.3])
    model = linearize(record(vector), np.eye(72))
    h = 1e-6
    numerical = np.column_stack(
        [
            (
                linearize(record(vector + h * direction), np.eye(72))["residual"][:3]
                - linearize(record(vector - h * direction), np.eye(72))["residual"][:3]
            )
            / (2 * h)
            for direction in np.eye(3)
        ]
    )
    assert_allclose(numerical, model["projection_blocks"][0], atol=2e-9, rtol=2e-8)


def test_rounded_sqrt_cannot_give_zero_merit_to_an_outside_cap():
    q = np.zeros(24)
    q[0] = np.nextafter(1.0, 2.0)
    assert np.sqrt(q[0]) == 1
    assert cap_loss(q) > 0
    q[0] = 1
    assert cap_loss(q) == 0
    physical = record([1, 1e-8, 0])
    assert physical["normalized_squared_errors"][0, 0] > 1
    assert not physical["feasible"]
    assert linearize(physical, np.eye(72))["loss"] > 0


def test_raw_derivative_is_preserved_for_a_later_reactivated_group():
    raw = np.eye(72)
    assert not linearize(record([0.9, 0, 0]), raw)["jacobian"].any()
    assert linearize(record([1.1, 0, 0]), raw)["jacobian"][0, 0] > 0
    assert_array_equal(raw, np.eye(72))
    raw[0, 6] = 1e-20
    with pytest.raises(ValueError, match="Future command"):
        linearize(record([0.9, 0, 0]), raw)


def test_restore_qp_and_sumq_diagnostics_are_unchanged():
    value = inputs()
    anchor = value["historical_initial"]
    args = (
        anchor,
        value["Js"],
        value["Jq"],
        value["Je"],
        value["previous"],
        -np.ones((12, 6)),
        np.ones((12, 6)),
        np.full(24, 3.0),
        0.005,
        "restore",
        value["config"],
    )
    original, observed = build_stage_problem(*args), CapDistanceObjective.problem(*args)
    assert set(observed) == set(original)
    for key in ["B", "g", "A", "b"]:
        assert_array_equal(original[key], observed[key])
    old = metrics(anchor, value["config"])
    new = CapDistanceObjective.metrics(anchor, value["config"])
    assert all(new[key] == val for key, val in old.items())


def test_pose_prediction_is_expanded_undamped_gn_in_cap_loss_units():
    value = inputs()
    anchor, policy = value["historical_initial"], CapDistanceObjective()
    problem = policy.problem(
        anchor,
        value["Js"],
        value["Jq"],
        value["Je"],
        value["previous"],
        -np.ones((12, 6)),
        np.ones((12, 6)),
        np.full(24, 3.0),
        0.005,
        "pose",
        value["config"],
    )
    before = policy.metrics(anchor, value["config"])
    E = (anchor["tcp_errors"] / np.tile([0.01, 0.01, 0.01, 0.05, 0.05, 0.05], (12, 1))).ravel()
    step = np.zeros(72)
    step[0] = 0.001
    _, reduction = policy.prediction(
        before,
        anchor["safety_slack"],
        anchor["normalized_squared_errors"].ravel(),
        E,
        value["Js"],
        value["Jq"],
        value["Je"],
        step,
        value["config"],
        "pose",
        problem,
    )
    assert_allclose(reduction, 0.00495, atol=1e-16, rtol=0)
    damped = reduction - 0.5 * (step @ step)
    assert reduction - damped > 4.9e-7
    _, zero = policy.prediction(
        before,
        anchor["safety_slack"],
        anchor["normalized_squared_errors"].ravel(),
        E,
        value["Js"],
        value["Jq"],
        value["Je"],
        np.zeros(72),
        value["config"],
        "pose",
        problem,
    )
    assert zero == 0
