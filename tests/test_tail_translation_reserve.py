"""Analytic fixed-H12 probes; no response model, simulator, or GPU is used."""

from copy import deepcopy

import numpy as np
import pytest

from s2a.geometry import pose_error
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.osc import OSCInterface
from s2a.tail_translation_reserve import SIGNED_STEPS, build_tail_translation_grid, probe_tail_translation_reserve


def synthetic_case(*, terminal=None, base=2e-8, prior_max_abs_residual=1e-6, prefix_error=0.0, full_witness=False):
    """Construct a complete saved record with a causal analytic tail response."""
    anchor, previous = np.zeros((12, 6)), np.zeros(6)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    lower, upper, profile = -2 * np.ones(7), 2 * np.ones(7), np.zeros((25, 7))
    desired, tool, normals = np.tile(np.eye(4), (12, 1, 1)), np.eye(4), np.eye(7)[6:]
    scalar = terminal if terminal is not None else lambda commands: base + 0.1 * commands[11, 0]

    def response(commands):
        endpoint_q, endpoint_velocity = np.zeros((12, 7)), np.zeros((12, 7))
        endpoint_velocity[:, 6] = -base
        endpoint_velocity[-1, 6] = -scalar(commands)
        inner_q = np.repeat(endpoint_q, 25, axis=0)
        inner_velocity = np.repeat(endpoint_velocity, 25, axis=0)
        poses, goals = desired.copy(), desired.copy()
        poses[:9, 0, 3] = prefix_error
        if not full_witness:
            poses[9:, 0, 3] = 0.02
        goals[-1, :3, 3] = commands[11, :3]
        periods = []
        for index in range(12):
            torque = np.zeros((25, 7))
            if index == 11:
                torque[:, :3] = commands[11, :3]
            periods.append(
                dict(
                    joint_position=inner_q[index * 25 : (index + 1) * 25].copy(),
                    joint_velocity=inner_velocity[index * 25 : (index + 1) * 25].copy(),
                    requested_torque=torque.copy(),
                    applied_torque=torque.copy(),
                    clipped=np.zeros((25, 7), dtype=bool),
                    mass_change=np.zeros(25),
                    jacobian_change=np.zeros(25),
                    goal=goals[index].copy(),
                    endpoint_pose=poses[index].copy(),
                    endpoint_twist=np.zeros(6),
                    update_matrices=True,
                    dynamics_queries=26,
                )
            )
        return dict(
            inner_joint_position=inner_q,
            inner_joint_velocity=inner_velocity,
            joint_position=endpoint_q,
            joint_velocity=endpoint_velocity,
            pose=poses,
            twist=np.zeros((12, 6)),
            goals=goals,
            periods=periods,
        )

    rollout = response(anchor)
    audit = audit_joint_preview_constraints(
        rollout, anchor, previous, interface, lower, upper, past_inner_error_rad=profile, terminal_normals=normals
    )
    errors = pose_error(desired, rollout["pose"] @ tool)
    squared = np.column_stack(
        (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    failed = np.flatnonzero(np.any(squared > 1, axis=1))
    prefix = int(failed[0]) if len(failed) else 12
    historical = dict(
        commands=anchor.copy(),
        rollout=rollout,
        audit=audit,
        tcp_errors=errors,
        normalized_squared_errors=squared,
        safety_slack=np.concatenate(
            [
                audit["groups"][key]["slack"].ravel()
                for key in ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")
            ]
        ),
        prefix_length=prefix,
        maximum_normalized_squared_error=float(np.max(squared)),
        maximum_pose_cap_violation=float(max(0, np.max(np.sqrt(squared)) - 1)),
        pose_loss=float(np.sum(squared)),
        feasible=bool(audit["feasible"] and prefix == 12),
    )
    return dict(
        predict=response,
        previous=previous,
        anchor=anchor,
        interface=interface,
        lower=lower,
        upper=upper,
        desired=desired,
        site_from_tcp=tool,
        past_inner_error_rad=profile,
        terminal_normals=normals,
        grid=build_tail_translation_grid(anchor, previous, interface.input_min, interface.input_max),
        historical_anchor=historical,
        prior_max_abs_residual=prior_max_abs_residual,
    )


def run_counted(arguments):
    original, commands = arguments["predict"], []

    def predict(command):
        commands.append(command.copy())
        return original(command)

    return probe_tail_translation_reserve(**dict(arguments, predict=predict)), commands


def test_grid_changes_only_the_fixed_tail_scalar_in_exact_order_and_retains_roundoff():
    anchor = np.zeros((12, 6))
    anchor[:, 5] = np.nextafter(np.nextafter(1.0, np.inf), np.inf)
    previous = anchor[0].copy()
    grid = build_tail_translation_grid(anchor, previous, -np.ones(6), np.ones(6))
    assert grid["admitted"] and grid["anchor_admission"]["box_upper"][0, 5] < 0
    keys = {anchor.tobytes()}
    for index, row in enumerate(grid["grid_commands"]):
        axis, step_index = divmod(index, 4)
        expected = anchor.copy()
        expected[11, axis] += SIGNED_STEPS[step_index]
        assert row["commands"].tobytes() == expected.tobytes()
        assert row["commands"][:11].tobytes() == anchor[:11].tobytes()
        assert row["commands"][11, 3:].tobytes() == anchor[11, 3:].tobytes()
        assert row["axis_index"] == axis and row["step_index"] == step_index
        assert row["command_key"] == expected.tobytes().hex()
        np.testing.assert_array_equal(row["actual_displacement"], expected - anchor)
        keys.add(expected.tobytes())
    assert len(keys) == 13


@pytest.mark.parametrize(
    "defect",
    ["one_illegal_candidate", "tampered_grid", "zero_displacement", "historical_command", "invalid_prior_residual"],
)
def test_complete_preflight_rejects_before_any_prediction(defect):
    arguments = synthetic_case()
    if defect == "one_illegal_candidate":
        arguments["anchor"][:, 0] = 0.9995
        arguments["previous"][0] = 0.9995
        arguments["grid"] = build_tail_translation_grid(
            arguments["anchor"], arguments["previous"], -np.ones(6), np.ones(6)
        )
        assert not arguments["grid"]["admitted"]
    elif defect == "tampered_grid":
        arguments["grid"]["grid_commands"][0]["commands"][10, 0] = 1e-6
    elif defect == "zero_displacement":
        arguments["anchor"][:] = 1e14
        arguments["previous"][:] = 1e14
        arguments["interface"] = OSCInterface(-1e15 * np.ones(6), 1e15 * np.ones(6), -np.ones(6), np.ones(6))
        arguments["grid"] = build_tail_translation_grid(
            arguments["anchor"],
            arguments["previous"],
            arguments["interface"].input_min,
            arguments["interface"].input_max,
        )
        assert all(row["zero_displacement"] for row in arguments["grid"]["grid_commands"])
    elif defect == "historical_command":
        arguments["historical_anchor"]["commands"][0, 0] = 1e-12
    else:
        arguments["prior_max_abs_residual"] = np.nan
    calls = []
    arguments["predict"] = lambda command: calls.append(command)
    with pytest.raises(ValueError):
        probe_tail_translation_reserve(**arguments)
    assert calls == []


def test_all_fourteen_calls_include_paid_repeat_and_reserve_does_not_replace_full_witness():
    result, calls = run_counted(synthetic_case())
    assert len(calls) == result["response_evaluations"] == 14
    assert len({command.tobytes() for command in calls}) == result["unique_command_count"] == 13
    assert calls[0].tobytes() == calls[-1].tobytes()
    assert result["selected_response_index"] == 1
    assert result["improvement_found"] and result["sufficient_reserve_restored"]
    assert result["selected"]["prefix_length"] == 9 and not result["selected"]["feasible"]
    assert result["selected_summary"]["terminal"] == pytest.approx(0.00010002)
    assert result["selected_summary"]["tcp_errors"].shape == (12, 6)
    assert result["causal_checks_all_passed"] and result["repeatability"]["exact"]
    assert not result["control_qualified"] and result["total_iterations"] == 0
    for index, row in enumerate(result["request_ledger"]):
        assert row["request_index"] == row["response_index"] == row["calls_before"] == index
        assert row["calls_after"] == index + 1 and row["charged"]
    assert result["request_ledger"][-1]["outcome"] == "repeated"
    assert result["request_ledger"][-1]["repeat_of"] == 0


@pytest.mark.parametrize(
    "defect", ["period_mass_summary", "inner_velocity", "endpoint_twist", "top_goal_rotation", "period_goal_rotation"]
)
def test_tail_input_cannot_change_saved_prefix_or_either_goal_rotation_representation(defect):
    arguments = synthetic_case()
    original = arguments["predict"]

    def predict(command):
        response = original(command)
        if command[11, 0] == 0.001:
            if defect == "period_mass_summary":
                response["periods"][10]["mass_change"][0] = 1e-15
            elif defect == "inner_velocity":
                response["inner_joint_velocity"][274, 0] = 1e-15
            elif defect == "endpoint_twist":
                response["twist"][10, 0] = 1e-15
            else:
                angle = 1e-4
                rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                if defect == "top_goal_rotation":
                    response["goals"][11, :3, :3] = rotation
                else:
                    response["periods"][11]["goal"][:3, :3] = rotation
        return response

    result, calls = run_counted(dict(arguments, predict=predict))
    assert len(calls) == 14 and not result["causal_checks"][1]["passed"]
    assert not result["candidate_checks"][0]["restoration_eligible"]
    expected_reason = "rotation_goals" if "rotation" in defect else "causal_prefix"
    assert result["candidate_checks"][0]["rejection_reasons"] == [expected_reason]
    assert result["selected_response_index"] == 3  # Only the corrupted candidate is disqualified.
    assert result["causal_checks"][3]["passed"]


@pytest.mark.parametrize("defect", ["historical", "repeat"])
def test_global_exact_failure_preserves_every_response_but_disqualifies_restoration(defect):
    arguments = synthetic_case()
    if defect == "historical":
        arguments["historical_anchor"]["rollout"]["twist"][11, 0] = 1e-15
    original, count = arguments["predict"], 0

    def predict(command):
        nonlocal count
        count += 1
        response = original(command)
        if defect == "repeat" and count == 14:
            response["twist"][11, 0] = 1e-15
        return response

    result = probe_tail_translation_reserve(**dict(arguments, predict=predict))
    assert count == 14 and result["selected"] is None and not result["improvement_found"]
    assert not result["sufficient_reserve_restored"] and result["eligible_ranking"] == []
    assert all(not row["restoration_eligible"] for row in result["candidate_checks"])


def test_tail_clipping_change_is_reported_without_an_unregistered_global_clipping_gate():
    arguments = synthetic_case()
    original = arguments["predict"]

    def predict(command):
        response = original(command)
        if command[11, 0] == 0.001:
            response["periods"][11]["clipped"][0, 0] = True
        return response

    result = probe_tail_translation_reserve(**dict(arguments, predict=predict))
    assert result["clipping_checks"][1]["changed_count"] == 1
    assert result["selected_response_index"] == 1 and result["candidate_checks"][0]["restoration_eligible"]


def test_saved_goal_roundoff_is_compared_exactly_without_new_rotation_admission():
    arguments = synthetic_case()
    original = arguments["predict"]

    def save_rounded_goal(rollout):
        value = float(np.float32(1.0) + np.float32(2e-7))
        rollout["goals"][:, 0, 0] = value
        for period in rollout["periods"]:
            period["goal"][0, 0] = value
        return rollout

    save_rounded_goal(arguments["historical_anchor"]["rollout"])
    arguments["predict"] = lambda command: save_rounded_goal(original(command))
    result = probe_tail_translation_reserve(**arguments)
    assert result["causal_checks_all_passed"] and result["historical_anchor_comparison"]["comparison"]["exact"]
    assert result["selected_response_index"] == 1


def test_exact_ties_use_actual_displacement_then_fixed_request_order_without_approximate_terminal_ties():
    base, restored = 2e-8, 1e-4

    def terminal(command):
        return restored if np.any(command != 0) else base

    result = probe_tail_translation_reserve(**synthetic_case(terminal=terminal))
    assert result["eligible_ranking"] == [3, 4, 7, 8, 11, 12, 1, 2, 5, 6, 9, 10]
    assert result["selected_response_index"] == 3

    def slightly_better(command):
        return np.nextafter(restored, np.inf) if command[11, 1] == 0.001 else terminal(command)

    result = probe_tail_translation_reserve(**synthetic_case(terminal=slightly_better))
    assert result["selected_response_index"] == 5


def test_no_strict_improvement_keeps_eligible_ranking_without_inventing_a_selection():
    result, calls = run_counted(synthetic_case(terminal=lambda command: 2e-8))
    assert len(calls) == 14 and len(result["eligible_ranking"]) == 12
    assert result["selected_response_index"] is result["selected"] is result["selected_summary"] is None
    assert not result["improvement_found"] and not result["sufficient_reserve_restored"]


def test_safety_buffer_is_strict_even_when_original_joint_audit_allows_roundoff():
    def terminal(command):
        if command[11, 0] == 0.001:
            return 1e-8
        return np.nextafter(1e-8, -np.inf) if np.any(command) else 5e-9

    result = probe_tail_translation_reserve(**synthetic_case(terminal=terminal, base=5e-9))
    assert all(entry["audit"]["feasible"] for entry in result["evaluated_candidates"])
    assert result["eligible_ranking"] == [1] and result["selected_response_index"] == 1
    assert result["candidate_checks"][0]["safety_buffer_satisfied"]
    assert not result["candidate_checks"][2]["safety_buffer_satisfied"]
    assert not result["sufficient_reserve_restored"]


@pytest.mark.parametrize("error,accepted", [(0.01, True), (np.nextafter(0.01, np.inf), False)])
def test_preserved_prefix_uses_original_strict_pose_caps(error, accepted):
    result = probe_tail_translation_reserve(**synthetic_case(prefix_error=error))
    assert result["candidate_checks"][0]["prefix9_preserved"] is accepted
    assert result["improvement_found"] is accepted
    assert result["causal_checks_all_passed"]


def test_selection_is_independent_of_the_prior_reserve_target_and_witnesses_do_not_stop_calls():
    result, calls = run_counted(synthetic_case(prior_max_abs_residual=1e-3, full_witness=True))
    assert len(calls) == 14 and result["evaluated_candidates"][0]["feasible"]
    assert result["selected_response_index"] == 1 and result["selected"]["feasible"]
    assert result["improvement_found"] and not result["sufficient_reserve_restored"]
    assert not result["control_qualified"]
    saved = deepcopy(result["evaluated_candidates"][1]["commands"])
    result["selected"]["commands"][0, 0] = 0.2
    np.testing.assert_array_equal(result["evaluated_candidates"][1]["commands"], saved)
