"""Complete analytic H12 responses exercise finite, causal tail repair."""

from copy import deepcopy

import numpy as np
import pytest
from test_tail_translation_reserve import synthetic_case

from s2a.geometry import pose_error
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.tail_translation_reserve import SAFETY_NAMES, build_tail_translation_grid
from s2a.terminal_only_tail_repair import ENTRY_FIELDS, probe_terminal_only_tail_repair


def case(**kwargs):
    return synthetic_case(prior_max_abs_residual=None, **kwargs)


def run_counted(arguments):
    original, calls = arguments["predict"], []

    def predict(command):
        calls.append(command.copy())
        return original(command)

    return probe_terminal_only_tail_repair(**dict(arguments, predict=predict)), calls


def refresh_history(arguments):
    """Build the saved full entry after a coherent synthetic response change."""
    history = arguments["historical_anchor"]
    rollout = arguments["predict"](arguments["anchor"])
    audit = audit_joint_preview_constraints(
        rollout,
        arguments["anchor"],
        arguments["previous"],
        arguments["interface"],
        arguments["lower"],
        arguments["upper"],
        past_inner_error_rad=arguments["past_inner_error_rad"],
        terminal_normals=arguments["terminal_normals"],
    )
    errors = pose_error(arguments["desired"], rollout["pose"] @ arguments["site_from_tcp"])
    normalized = np.column_stack(
        (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    failures = np.flatnonzero(np.any(normalized > 1, axis=1))
    prefix = int(failures[0]) if len(failures) else 12
    history.update(
        rollout=rollout,
        audit=audit,
        tcp_errors=errors,
        normalized_squared_errors=normalized,
        safety_slack=np.concatenate([audit["groups"][key]["slack"].ravel() for key in SAFETY_NAMES]),
        prefix_length=prefix,
        maximum_normalized_squared_error=float(np.max(normalized)),
        maximum_pose_cap_violation=float(max(0, np.max(np.sqrt(normalized)) - 1)),
        pose_loss=float(np.sum(normalized)),
        feasible=bool(audit["feasible"] and prefix == 12),
    )


@pytest.mark.parametrize("defect", ["one_illegal", "tampered_grid", "command_identity", "missing_history", "bound"])
def test_all_input_admission_precedes_any_response(defect):
    arguments = case()
    if defect == "one_illegal":
        arguments["anchor"][:, 0] = arguments["previous"][0] = 0.9995
        arguments["grid"] = build_tail_translation_grid(
            arguments["anchor"], arguments["previous"], -np.ones(6), np.ones(6)
        )
        assert not arguments["grid"]["admitted"]
    elif defect == "tampered_grid":
        arguments["grid"]["grid_commands"][0]["commands"][10, 0] = 1e-6
    elif defect == "command_identity":
        arguments["historical_anchor"]["commands"][0, 0] = 1e-12
    elif defect == "missing_history":
        del arguments["historical_anchor"]["pose_loss"]
    else:
        arguments["prior_max_abs_residual"] = 0.0
    calls = []
    arguments["predict"] = lambda command: calls.append(command)
    with pytest.raises(ValueError):
        probe_terminal_only_tail_repair(**arguments)
    assert not calls


def test_full_fourteen_calls_keep_full_witness_separate_from_terminal_repair():
    result, calls = run_counted(case(base=-1e-6))
    assert len(calls) == result["response_evaluations"] == result["planned_request_count"] == 14
    assert len({command.tobytes() for command in calls}) == result["unique_command_count"] == 13
    assert calls[0].tobytes() == calls[-1].tobytes()
    assert result["total_iterations"] == 0
    assert result["selected_response_index"] == 1 and result["improvement_found"]
    assert result["selected"]["prefix_length"] == 9 and not result["selected"]["feasible"]
    assert result["selected_summary"]["terminal"] == pytest.approx(0.000099)
    assert not result["empirical_error_bound_used"] and not result["control_qualified"]
    assert not ({"sufficient_reserve_restored", "reserve_target", "prior_max_abs_residual"} & result.keys())
    assert set(ENTRY_FIELDS) == set(result["evaluated_candidates"][0])
    assert result["causal_checks_all_passed"] and result["repeatability"]["exact"]
    for index, row in enumerate(result["request_ledger"]):
        assert row["request_index"] == row["response_index"] == row["calls_before"] == index
        assert row["calls_after"] == index + 1 and row["charged"]
        assert row["commands"].tobytes() == calls[index].tobytes()
    assert result["request_ledger"][-1]["repeat_of"] == 0
    for row in result["causal_checks"]:
        assert (row["prefix_period_count"], row["prefix_inner_count"], row["prefix_endpoint_count"]) == (11, 275, 11)
    saved = deepcopy(result["evaluated_candidates"][1]["commands"])
    result["selected"]["commands"][0, 0] = 0.2
    np.testing.assert_array_equal(result["evaluated_candidates"][1]["commands"], saved)


def test_improved_terminal_cannot_mask_an_earlier_braking_failure():
    arguments = case(base=-1e-6)
    original = arguments["predict"]

    def predict(command):
        response = original(command)
        response["joint_velocity"][8, 0] = 20.0
        response["inner_joint_velocity"][200:225, 0] = 20.0
        response["periods"][8]["joint_velocity"][:, 0] = 20.0
        return response

    arguments["predict"] = predict
    refresh_history(arguments)
    result, calls = run_counted(arguments)
    assert len(calls) == 14 and result["causal_checks_all_passed"]
    assert result["historical_anchor_comparison"]["comparison"]["exact"]
    assert result["terminal_values"][1] > 1e-8
    assert result["evaluated_candidates"][1]["audit"]["groups"]["braking_upper"]["slack"][8, 0] < 0
    assert result["selected"] is None and not result["improvement_found"]
    assert all(not row["repair_eligible"] for row in result["candidate_checks"])
    assert result["candidate_checks"][0]["rejection_reasons"] == ["joint_audit", "safety_buffer"]


def test_optimizer_metadata_is_recorded_but_all_generated_history_fields_are_compared():
    arguments = case()
    arguments["historical_anchor"].update(stage_index=50, alpha=0.125, optimizer_diagnostic={"arbitrary": 3})
    result = probe_terminal_only_tail_repair(**arguments)
    history = result["historical_anchor_comparison"]
    assert history["compared_fields"] == list(ENTRY_FIELDS)
    assert history["ignored_historical_fields"] == ["alpha", "optimizer_diagnostic", "stage_index"]
    assert history["comparison"]["exact"] and result["improvement_found"]
    assert "audit.groups.terminal.slack" in history["comparison"]["arrays"]
    assert "rollout.inner_joint_position" in history["comparison"]["arrays"]


@pytest.mark.parametrize("defect", ["historical_rollout", "historical_audit", "repeat"])
def test_historical_or_repeat_disagreement_disqualifies_every_candidate_after_all_calls(defect):
    arguments = case()
    if defect == "historical_rollout":
        arguments["historical_anchor"]["rollout"]["twist"][11, 0] = 1e-15
    elif defect == "historical_audit":
        arguments["historical_anchor"]["audit"]["groups"]["inner_lower"]["minimum"] += 1e-12
    original, count = arguments["predict"], 0

    def predict(command):
        nonlocal count
        count += 1
        response = original(command)
        if defect == "repeat" and count == 14:
            response["twist"][11, 0] = 1e-15
        return response

    result = probe_terminal_only_tail_repair(**dict(arguments, predict=predict))
    assert count == 14 and result["selected"] is None and not result["improvement_found"]
    assert result["eligible_ranking"] == []
    assert all(not row["repair_eligible"] for row in result["candidate_checks"])


@pytest.mark.parametrize("defect", ["period", "inner", "endpoint", "top_rotation", "period_rotation"])
def test_prefix_or_rotation_changes_disqualify_only_the_corrupt_candidate(defect):
    arguments = case()
    original = arguments["predict"]

    def predict(command):
        response = original(command)
        if command[11, 0] == 0.001:
            if defect == "period":
                response["periods"][10]["mass_change"][0] = 1e-15
            elif defect == "inner":
                response["inner_joint_velocity"][274, 0] = 1e-15
            elif defect == "endpoint":
                response["twist"][10, 0] = 1e-15
            else:
                key = response["goals"][11] if defect == "top_rotation" else response["periods"][11]["goal"]
                key[0, 0] = np.nextafter(1.0, np.inf)
        return response

    result, calls = run_counted(dict(arguments, predict=predict))
    assert len(calls) == 14 and not result["candidate_checks"][0]["repair_eligible"]
    assert result["selected_response_index"] == 3
    reason = "rotation_goals" if "rotation" in defect else "causal_prefix"
    assert result["candidate_checks"][0]["rejection_reasons"] == [reason]


def test_rank_uses_exact_terminal_then_actual_displacement_then_request_index():
    base, improved = 2e-8, 1e-4

    def terminal(command):
        return improved if np.any(command) else base

    result = probe_terminal_only_tail_repair(**case(terminal=terminal))
    assert result["eligible_ranking"] == [3, 4, 7, 8, 11, 12, 1, 2, 5, 6, 9, 10]
    assert result["selected_response_index"] == 3

    def tiny_improvement(command):
        return np.nextafter(improved, np.inf) if command[11, 1] == 0.001 else terminal(command)

    result = probe_terminal_only_tail_repair(**case(terminal=tiny_improvement))
    assert result["selected_response_index"] == 5


def test_no_strict_terminal_improvement_keeps_the_ranking_but_no_selection():
    result, calls = run_counted(case(terminal=lambda command: 2e-8, full_witness=True))
    assert len(calls) == 14 and result["evaluated_candidates"][0]["feasible"]
    assert len(result["eligible_ranking"]) == 12
    assert result["selected"] is result["selected_response_index"] is result["selected_summary"] is None
    assert not result["improvement_found"] and not result["control_qualified"]


def test_original_audit_tolerance_does_not_replace_the_strict_buffer():
    def terminal(command):
        if command[11, 0] == 0.001:
            return 1e-8
        return np.nextafter(1e-8, -np.inf) if np.any(command) else 5e-9

    result = probe_terminal_only_tail_repair(**case(terminal=terminal, base=5e-9))
    assert all(row["audit"]["feasible"] for row in result["evaluated_candidates"])
    assert result["eligible_ranking"] == [1] and result["selected_response_index"] == 1


@pytest.mark.parametrize("error,accepted", [(0.01, True), (np.nextafter(0.01, np.inf), False)])
def test_prefix_retains_original_position_cap_without_tolerance(error, accepted):
    result = probe_terminal_only_tail_repair(**case(prefix_error=error))
    assert result["candidate_checks"][0]["prefix9_preserved"] is accepted
    assert result["improvement_found"] is accepted


def test_rotation_cap_is_recomputed_from_the_complete_pose_response():
    arguments = case()
    original = arguments["predict"]
    angle = 0.050001
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    def predict(command):
        response = original(command)
        response["pose"][8, :3, :3] = rotation
        response["periods"][8]["endpoint_pose"][:3, :3] = rotation
        return response

    arguments["predict"] = predict
    refresh_history(arguments)
    result = probe_terminal_only_tail_repair(**arguments)
    assert result["causal_checks_all_passed"]
    assert result["evaluated_candidates"][1]["normalized_squared_errors"][8, 1] > 1.0
    assert result["evaluated_candidates"][1]["prefix_length"] == 8
    assert result["selected"] is None and not result["improvement_found"]
