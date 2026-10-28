"""Finite analytic secant diagnostics; no robot, simulator or model is loaded."""

from copy import deepcopy

import numpy as np
import pytest

from s2a.geometry import pose_error
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.osc import OSCInterface
from s2a.terminal_secant_probe import (
    FIT_STEP_SIZES,
    STEP_SIZES,
    build_terminal_secant_geometry,
    fit_terminal_secant_row,
    probe_terminal_secant,
)


def geometry_inputs():
    return dict(
        anchor_commands=np.zeros((1, 6)),
        previous_command=np.zeros(6),
        input_min=-np.ones(6),
        input_max=np.ones(6),
        source_unit_directions=np.eye(6)[:3].reshape(3, 1, 6),
    )


def synthetic_case(*, base=0.02, curvature=0.0, orthogonal_slope=0.0):
    arguments = geometry_inputs()
    geometry = build_terminal_secant_geometry(**arguments)
    unit = arguments["source_unit_directions"][0]
    orthogonal = geometry["v"].ravel()
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    lower, upper = -2 * np.ones(7), 2 * np.ones(7)
    desired, tool, profile = np.eye(4)[None], np.eye(4), np.zeros((25, 7))
    normals = np.eye(7)[6:]

    def scalar(commands):
        coordinate = commands[0, 0]
        return base + 0.4 * coordinate + curvature * coordinate**2 + orthogonal_slope * (orthogonal @ commands.ravel())

    def response(commands):
        velocity = np.zeros((1, 7))
        velocity[-1, 6] = -scalar(commands)
        return dict(
            inner_joint_position=np.zeros((25, 7)),
            inner_joint_velocity=np.repeat(velocity, 25, axis=0),
            joint_position=np.zeros((1, 7)),
            joint_velocity=velocity,
            pose=desired.copy(),
            twist=np.zeros((1, 6)),
            goals=desired.copy(),
            periods=[dict(clipped=np.zeros((25, 7), dtype=bool), goal=tool.copy())],
        )

    anchor = arguments["anchor_commands"]
    rollout = response(anchor)
    audit = audit_joint_preview_constraints(
        rollout,
        anchor,
        arguments["previous_command"],
        interface,
        lower,
        upper,
        past_inner_error_rad=profile,
        terminal_normals=normals,
    )
    errors = pose_error(desired, rollout["pose"] @ tool)
    historical = dict(
        commands=anchor.copy(),
        rollout=rollout,
        audit=audit,
        tcp_errors=errors,
        normalized_squared_errors=np.zeros((1, 2)),
        safety_slack=np.concatenate(
            [
                audit["groups"][name]["slack"].ravel()
                for name in ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")
            ]
        ),
        prefix_length=1,
        maximum_normalized_squared_error=0.0,
        maximum_pose_cap_violation=0.0,
        pose_loss=0.0,
        feasible=True,
    )
    commands = np.asarray([anchor + step * unit for step in FIT_STEP_SIZES])
    fit = fit_terminal_secant_row(anchor, unit, np.zeros(6), base, commands, [scalar(command) for command in commands])
    return dict(
        predict=response,
        previous=arguments["previous_command"],
        anchor=anchor,
        interface=interface,
        lower=lower,
        upper=upper,
        desired=desired,
        site_from_tcp=tool,
        past_inner_error_rad=profile,
        terminal_normals=normals,
        geometry=geometry,
        fit=fit,
        historical_anchor=historical,
    )


def test_projected_gaussian_has_independent_direction_and_forty_five_degree_mixtures():
    arguments = geometry_inputs()
    result = build_terminal_secant_geometry(**arguments)
    # With source rows e0,e1,e2 and no guard rows, the projection has an exact
    # independent formula: drop the first three Gaussian coordinates.
    expected = np.random.default_rng(42).standard_normal(6)
    expected[:3] = 0
    expected /= np.linalg.norm(expected)
    if expected[3] < 0:
        expected *= -1
    np.testing.assert_allclose(result["v"].ravel(), expected, rtol=0, atol=1e-15)
    assert result["projection_rank"] == result["nullspace_dimension"] == 3
    assert result["sign_index"] == 3 and result["projection_residual_inf"] <= 1e-10
    assert result["admitted"] and len(result["grid_commands"]) == 15
    np.testing.assert_allclose(result["source_direction_cosine"][:, 0], [0, 2**-0.5, 2**-0.5], atol=1e-15)
    assert len({row["command_key"] for row in result["grid_commands"]}) == 15


def test_original_bound_roundoff_is_retained_without_clipping_or_new_direction_selection():
    arguments = geometry_inputs()
    value = 1.0
    for _ in range(3):
        value = np.nextafter(value, np.inf)
    arguments["anchor_commands"][0, 0] = value
    arguments["previous_command"][0] = value
    arguments["source_unit_directions"][0, 0, 0] = -1
    result = build_terminal_secant_geometry(**arguments)
    assert result["admitted"]
    assert result["anchor_commands"][0, 0] == value > 1
    rows = [row for row in result["projection_row_metadata"] if row["kind"] == "command_guard"]
    assert [(row["group"], row["axis"]) for row in rows] == [("box_upper", 0)]
    np.testing.assert_allclose(result["projection_rows"] @ result["v"].ravel(), 0, atol=1e-14)


def test_rate_guard_uses_period_difference_rows_and_an_invalid_grid_is_retained():
    arguments = geometry_inputs()
    arguments["anchor_commands"] = np.zeros((2, 6))
    arguments["anchor_commands"][1, 3] = 0.2495
    sources = np.zeros((3, 2, 6))
    sources[0, 1, 3] = 1
    sources[1, 0, 0] = sources[2, 0, 1] = 1
    arguments["source_unit_directions"] = sources
    result = build_terminal_secant_geometry(**arguments)
    index = next(
        i
        for i, row in enumerate(result["projection_row_metadata"])
        if row.get("group") == "rate_upper" and row["axis"] == 9
    )
    expected = np.zeros(12)
    expected[3], expected[9] = 1, -1
    np.testing.assert_array_equal(result["projection_rows"][index], expected)
    assert not result["admitted"]
    assert any("rate_upper" in row["admission"]["reasons"] for row in result["grid_commands"])
    assert len(result["grid_commands"]) == 15


def test_exhausted_nullspace_fails_without_drawing_replacement_directions():
    arguments = geometry_inputs()
    arguments["anchor_commands"][:] = 1
    arguments["previous_command"][:] = 1
    with pytest.raises(ValueError, match="exhaust.*nullspace"):
        build_terminal_secant_geometry(**arguments)


def test_rank_one_fit_uses_actual_rounded_displacements_not_nominal_step_lengths():
    anchor = np.full((1, 6), 1e8)
    unit = np.eye(6)[0].reshape(1, 6)
    commands = np.asarray([anchor + h * unit for h in FIT_STEP_SIZES])
    delta = commands[:, 0, 0] - anchor[0, 0]
    old_row = unit.ravel()
    result = fit_terminal_secant_row(anchor, unit, old_row, 2.0, commands, 2.0 + 3.0 * delta)
    assert result["coefficient"] == pytest.approx(2.0, abs=1e-10)
    np.testing.assert_allclose(result["corrected_terminal_row"], 3 * unit.ravel(), rtol=0, atol=1e-10)
    np.testing.assert_array_equal(result["source_coordinates"], delta)
    nominal_fit = np.dot(FIT_STEP_SIZES, 2 * delta) / np.dot(FIT_STEP_SIZES, FIT_STEP_SIZES)
    assert abs(nominal_fit - result["coefficient"]) > 1e-7
    changed = commands.copy()
    changed[0, 0, 1] = np.nextafter(changed[0, 0, 1], np.inf)
    with pytest.raises(ValueError, match="Exact positive source commands"):
        fit_terminal_secant_row(anchor, unit, old_row, 2.0, changed, 2.0 + 3.0 * delta)


def test_affine_terminal_row_recovers_on_unfitted_directions_with_exact_seventeen_calls():
    arguments = synthetic_case()
    response, calls = arguments["predict"], []

    def predict(commands):
        calls.append(commands.copy())
        return response(commands)

    arguments["predict"] = predict
    result = probe_terminal_secant(**arguments)
    assert result["extrapolation_passed"] and result["safety_accuracy_passed"]
    assert result["response_evaluations"] == len(calls) == 17
    assert result["unique_command_count"] == 16 and result["total_iterations"] == 0
    assert result["evaluated_candidates"][0]["feasible"]  # A witness must not truncate the fixed grid.
    assert calls[0].tobytes() == calls[-1].tobytes()
    assert len({command.tobytes() for command in calls[:-1]}) == 16
    assert [row["step"] for row in result["request_ledger"][1:-1]] == list(STEP_SIZES) * 3
    assert all(row["charged"] and row["calls_after"] == row["calls_before"] + 1 for row in result["request_ledger"])
    assert result["request_ledger"][-1]["outcome"] == "repeated"
    assert result["repeatability"]["exact"] and result["historical_anchor_comparison"]["comparison"]["exact"]
    assert not result["control_qualified"]
    for record, request in zip(result["predictions"], result["request_ledger"][1:-1], strict=True):
        expected = 0.4 * request["actual_displacement"][0, 0]
        assert record["corrected_predicted_change"] == pytest.approx(expected, abs=1e-16)
        assert abs(record["corrected_residual"]) < 1e-15


def test_unseen_orthogonal_component_can_falsify_the_training_direction_correction():
    result = probe_terminal_secant(**synthetic_case(orthogonal_slope=1.0))
    assert not result["extrapolation_passed"]
    assert result["qualification"]["pure_direction_check"]["passed"]
    assert any(not row["passed"] for row in result["qualification"]["mixed_direction_checks"])
    assert result["response_evaluations"] == 17  # Report the entire held-out grid despite failed extrapolation.


def test_relative_extrapolation_improvement_is_separate_from_absolute_safety_accuracy():
    result = probe_terminal_secant(**synthetic_case(base=1.001e-8, curvature=0.01))
    assert result["extrapolation_passed"]
    assert not result["safety_accuracy_passed"]
    check = result["qualification"]
    assert check["max_abs_corrected_residual"] > check["safety_available_margin"] > 0
    assert not result["control_qualified"]


@pytest.mark.parametrize("defect", ["geometry", "fit", "grid_admission"])
def test_all_inputs_and_fifteen_commands_are_checked_before_any_prediction(defect):
    arguments = synthetic_case()
    calls = []
    arguments["predict"] = lambda commands: calls.append(commands)
    if defect == "geometry":
        arguments["geometry"]["grid_commands"][-1]["commands"][0, 0] += 1e-9
    elif defect == "fit":
        arguments["fit"]["corrected_terminal_row"][0] += 1e-9
    else:
        new_inputs = geometry_inputs()
        new_inputs["anchor_commands"][0, 0] = 0.2495
        geometry = build_terminal_secant_geometry(**new_inputs)
        assert not geometry["admitted"]
        arguments["anchor"] = new_inputs["anchor_commands"]
        arguments["geometry"] = geometry
    with pytest.raises(ValueError):
        probe_terminal_secant(**arguments)
    assert calls == []


@pytest.mark.parametrize("defect", ["clipping", "repeat", "historical"])
def test_complete_repeat_and_clipping_gates_reject_numeric_success(defect):
    arguments = synthetic_case()
    response = arguments["predict"]
    calls = 0

    def predict(commands):
        nonlocal calls
        calls += 1
        value = response(commands)
        if defect == "clipping" and calls == 2:
            value["periods"][0]["clipped"][0, 0] = True
        elif defect == "repeat" and calls == 17:
            value["twist"][0, 0] = 1e-8
        return value

    arguments["predict"] = predict
    if defect == "historical":
        arguments["historical_anchor"] = deepcopy(arguments["historical_anchor"])
        arguments["historical_anchor"]["rollout"]["twist"][0, 0] = 1e-8
    result = probe_terminal_secant(**arguments)
    assert calls == 17 and not result["extrapolation_passed"]
    assert all(row["passed"] for row in result["qualification"]["mixed_direction_checks"])
    gate = {"clipping": "all_clipping_equal", "repeat": "repeat_exact", "historical": "historical_exact"}[defect]
    assert not result["qualification"][gate]
