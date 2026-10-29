"""Diagnose which preview period and hard constraint reject a frozen plan."""

from copy import deepcopy

import numpy as np
import pytest

from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.osc import OSCInterface


def example(horizon=2):
    return dict(
        rollout=dict(
            inner_joint_position=np.zeros((25 * horizon, 2)),
            joint_position=np.zeros((horizon, 2)),
            joint_velocity=np.zeros((horizon, 2)),
        ),
        commands=np.zeros((horizon, 6)),
        previous=np.zeros(6),
        interface=OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6)),
        lower=np.array([-1.0, -2.0]),
        upper=np.array([1.0, 2.0]),
        past_inner_error_rad=np.zeros((25, 2)),
        terminal_normals=np.empty((0, 2)),
    )


def test_first_and_future_inner_violations_have_distinct_locations_and_minima():
    inputs = example()
    inputs["rollout"]["inner_joint_position"][8, 0] = -0.97
    inputs["rollout"]["inner_joint_position"][31, 1] = 1.96
    result = audit_joint_preview_constraints(**inputs)
    assert not result["feasible"]
    assert result["horizon"] == 2 and result["joint_count"] == 2
    np.testing.assert_array_equal(result["groups"]["inner_lower"]["violated_indices"], [[8, 0]])
    np.testing.assert_array_equal(result["groups"]["inner_upper"]["violated_indices"], [[31, 1]])
    assert result["first_period_minima"]["inner_lower"] == pytest.approx(-0.02)
    assert result["future_period_minima"]["inner_lower"] == pytest.approx(0.95)
    assert result["first_period_minima"]["inner_upper"] == pytest.approx(0.95)
    assert result["future_period_minima"]["inner_upper"] == pytest.approx(-0.01)
    assert result["groups"]["braking_upper"]["minimum"] == pytest.approx(0.95)


def test_own_profile_tightens_first_inner_times_and_all_future_times_without_mutation():
    inputs = example()
    inputs["rollout"]["inner_joint_position"][:, 0] = 0.9
    assert audit_joint_preview_constraints(**inputs)["feasible"]
    inputs["past_inner_error_rad"][:, 0] = np.linspace(0, 0.2, 25)
    original = deepcopy(inputs)
    result = audit_joint_preview_constraints(**inputs)
    assert not result["feasible"]
    np.testing.assert_allclose(result["position_margin_rad"][:25, 0], np.linspace(0.05, 0.25, 25))
    np.testing.assert_allclose(result["position_margin_rad"][25:, 0], 0.25)
    np.testing.assert_allclose(result["position_margin_rad"][:, 1], 0.05)
    assert result["groups"]["inner_upper"]["slack"][0, 0] == pytest.approx(0.05)
    assert result["groups"]["inner_upper"]["slack"][24, 0] == pytest.approx(-0.15)
    assert result["future_period_minima"]["inner_upper"] == pytest.approx(-0.15)
    result["position_margin_rad"][:] = 100
    result["groups"]["inner_upper"]["slack"][:] = 100
    for name in ("commands", "previous", "lower", "upper", "past_inner_error_rad", "terminal_normals"):
        np.testing.assert_array_equal(inputs[name], original[name])
    for name in inputs["rollout"]:
        np.testing.assert_array_equal(inputs["rollout"][name], original["rollout"][name])


def test_command_box_and_both_rate_signs_include_previous_actual_command():
    inputs = example()
    inputs["interface"] = OSCInterface(-0.75 * np.ones(6), 0.75 * np.ones(6), -np.ones(6), np.ones(6))
    inputs["previous"][0] = -0.5
    inputs["commands"][:, 0] = [-0.76, -0.5]
    result = audit_joint_preview_constraints(**inputs)
    assert not result["feasible"]
    for group, expected in (("box_lower", [[0, 0]]), ("rate_lower", [[0, 0]]), ("rate_upper", [[1, 0]])):
        np.testing.assert_array_equal(result["groups"][group]["violated_indices"], expected)
        assert result["groups"][group]["minimum"] == pytest.approx(-0.01)
        assert result["groups"][group]["tolerance"] == 1e-12


def test_braking_uses_endpoint_velocity_and_terminal_normal_sign():
    inputs = example()
    inputs["rollout"]["joint_position"][:, 0] = [-0.94, 0.94]
    inputs["rollout"]["joint_velocity"][:] = [[-0.2, 0], [0.2, -0.4]]
    inputs["terminal_normals"] = [[1, 0], [-1, 0], [0, 1]]
    result = audit_joint_preview_constraints(**inputs)
    assert not result["feasible"]
    np.testing.assert_array_equal(result["groups"]["braking_lower"]["violated_indices"], [[0, 0]])
    np.testing.assert_array_equal(result["groups"]["braking_upper"]["violated_indices"], [[1, 0]])
    np.testing.assert_allclose(result["groups"]["terminal"]["slack"], [-0.2, 0.2, 0.4])
    np.testing.assert_array_equal(result["groups"]["terminal"]["violated_indices"], [[0]])
    assert result["first_period_minima"]["braking_lower"] == pytest.approx(-0.01)
    assert result["future_period_minima"]["braking_upper"] == pytest.approx(-0.01)
    assert "terminal" not in result["first_period_minima"]
    assert "terminal" not in result["future_period_minima"]
    shorter = audit_joint_preview_constraints(**inputs, braking_time_s=0.025)
    assert shorter["groups"]["braking_lower"]["minimum"] > 0
    assert shorter["groups"]["braking_upper"]["minimum"] > 0


def test_activity_and_violation_tolerances_differ_and_empty_terminal_survives_json():
    inputs = example(horizon=1)
    inputs["previous"][0] = 1
    inputs["commands"][0, 0] = 1 + 5e-13
    inputs["rollout"]["inner_joint_position"][0, 0] = 0.95 + 5e-11
    inputs["rollout"]["joint_position"][0, 0] = 0.95 + 5e-11
    inputs["terminal_normals"] = []
    result = audit_joint_preview_constraints(**inputs)
    assert result["feasible"] and result["activity_tolerance"] == 1e-8
    for group in ("box_upper", "inner_upper", "braking_upper"):
        np.testing.assert_array_equal(result["groups"][group]["active_indices"], [[0, 0]])
        assert not len(result["groups"][group]["violated_indices"])
    assert result["groups"]["inner_upper"]["tolerance"] == 1e-10
    assert result["groups"]["terminal"]["slack"].shape == (0,)
    assert result["groups"]["terminal"]["minimum"] is None
    assert result["groups"]["terminal"]["active_indices"].shape == (0, 1)
    assert all(value is None for value in result["future_period_minima"].values())
    inputs["commands"][0, 0] = 1 + 2e-12
    assert not audit_joint_preview_constraints(**inputs)["feasible"]
    inputs["commands"][0, 0] = 1
    inputs["rollout"]["inner_joint_position"][0, 0] = 0.95 + 2e-10
    assert not audit_joint_preview_constraints(**inputs)["feasible"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("commands", np.empty((0, 6))),
        ("previous", np.zeros(7)),
        ("upper", np.array([-1.0, 2.0])),
        ("past_inner_error_rad", -np.ones((25, 2))),
        ("past_inner_error_rad", np.linspace(0.1, 0, 50).reshape(25, 2)),
        ("terminal_normals", np.empty((0, 3))),
        ("braking_time_s", 0),
        ("braking_time_s", np.inf),
    ],
)
def test_rejects_ambiguous_constraint_inputs(field, value):
    inputs = example()
    inputs[field] = value
    with pytest.raises(ValueError):
        audit_joint_preview_constraints(**inputs)


@pytest.mark.parametrize("field", ["inner_joint_position", "joint_position", "joint_velocity"])
def test_rejects_missing_or_nonfinite_future_rollout(field):
    inputs = example()
    inputs["rollout"][field] = inputs["rollout"][field][:-1]
    with pytest.raises(ValueError):
        audit_joint_preview_constraints(**inputs)
    inputs = example()
    inputs["rollout"][field][-1, 0] = np.nan
    with pytest.raises(ValueError):
        audit_joint_preview_constraints(**inputs)
