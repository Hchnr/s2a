"""Check search accounting with toy responses, never native dynamics or physics.

The ledger helper consumes independently prevalidated classifications. Here we
recompute the synthetic response's pose and safety contract without importing
the production constraint audit; this is not a native dynamics qualification.
"""

import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_prefix_preserving_trajectory_plan import inputs, optimizer_result, response

import s2a.prefix_preserving_trajectory_plan as planning


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("check_tracking_v5_prefix_preserving_trajectory_control")
    return module.audit_prefix_search_ledger


def classify(answer, arguments):
    """Independently apply the toy pose, box/rate, inner/braking/terminal contract."""
    classifications = []
    for saved in answer["evaluated_candidates"]:
        commands = np.asarray(saved["commands"])
        rollout = arguments["predict"](commands.copy())
        actual = rollout["pose"] @ arguments["site_from_tcp"]
        desired = arguments["desired"]
        error = np.c_[
            desired[:, :3, 3] - actual[:, :3, 3],
            Rotation.from_matrix(desired[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1)).as_rotvec(),
        ]
        np.testing.assert_allclose(saved["tcp_errors"], error, rtol=0, atol=1e-13)
        normalized = np.column_stack(
            ((error[:, :3] ** 2).sum(axis=1) / 0.01**2, (error[:, 3:] ** 2).sum(axis=1) / 0.05**2)
        )
        failed = np.flatnonzero((normalized > 1).any(axis=1))
        prefix = int(failed[0]) if failed.size else len(commands)
        profile = arguments["past_inner_error_rad"]
        margin = np.tile(0.05 + profile.max(axis=0), (25 * len(commands), 1))
        margin[:25] = 0.05 + profile
        inner = rollout["inner_joint_position"]
        braking = rollout["joint_position"] + 0.1 * rollout["joint_velocity"]
        lower, upper = arguments["lower"], arguments["upper"]
        nonlinear = np.concatenate(
            [
                (inner - lower - margin).ravel(),
                (upper - margin - inner).ravel(),
                (braking - lower - 0.05).ravel(),
                (upper - 0.05 - braking).ravel(),
                -arguments["terminal_normals"] @ rollout["joint_velocity"][-1],
            ]
        )
        interface = arguments["interface"]
        changes = np.diff(np.vstack((arguments["previous"], commands)), axis=0)
        joint_feasible = bool(
            np.all(nonlinear >= -1e-10)
            and np.all(commands >= interface.input_min - 1e-12)
            and np.all(commands <= interface.input_max + 1e-12)
            and np.all(abs(changes) <= 0.25 + 1e-12)
        )
        classifications.append(
            dict(
                joint_feasible=joint_feasible,
                feasible=bool(joint_feasible and prefix == len(commands)),
                prefix_length=prefix,
                normalized_squared_errors=normalized,
                safety_min=float(nonlinear.min()),
            )
        )
    return classifications


@pytest.fixture
def two_phase_archive(monkeypatch):
    calls = []

    def advancing_return(_objective, start, **_kwargs):
        final = start.copy()
        final[6 if not calls else 12] = 0.1
        calls.append(final.copy())
        return optimizer_result(final, nit=2 if len(calls) == 1 else 1)

    monkeypatch.setattr(planning, "minimize", advancing_return)
    arguments = inputs(horizon=3)
    arguments["desired"][:, 0, 3] = [0, 0.1, 0.2]
    answer = planning.solve_prefix_preserving_trajectory_response(**arguments, max_iterations=5)
    assert len(calls) == 2
    return answer, classify(answer, arguments)


def test_real_slsqp_synthetic_archive_is_accepted(audit):
    arguments = inputs()
    arguments["desired"][:, 0, 3] = [0, 0.275]
    answer = planning.solve_prefix_preserving_trajectory_response(**arguments)
    checked = audit(answer, classify(answer, arguments))
    assert checked["selected_prefix_length"] == 2
    assert checked["termination_reason"] == "witness"
    assert checked["selected_trace_index"] in checked["witness_indices"]
    np.testing.assert_array_equal(
        answer["selected"]["commands"], answer["evaluated_candidates"][checked["selected_trace_index"]]["commands"]
    )


def test_normal_return_can_advance_between_shared_budget_phases(audit, two_phase_archive):
    answer, classifications = two_phase_archive
    checked = audit(answer, classifications)
    assert answer["response_evaluations"] == 3
    assert checked["total_iterations"] == 3
    assert checked["termination_reason"] == "witness"
    assert [p["max_iterations"] for p in answer["phases"]] == [5, 3]
    assert [p["prefix_before"] for p in checked["phase_checks"]] == [1, 2]
    assert [p["prefix_after"] for p in checked["phase_checks"]] == [2, 3]
    assert all(p["optimizer_returned"] and p["callback_count"] == 0 for p in answer["phases"])


def test_normal_return_final_evaluation_can_exhaust_response_budget(monkeypatch, audit):
    def fill_then_return(objective, start, *, callback, **_kwargs):
        callback(start)
        for number in range(1, 13):
            probe = start.copy()
            probe[7] = number * 1e-5
            objective(probe)
        final = start.copy()
        final[6] = 0.1
        return optimizer_result(final, nit=3)

    monkeypatch.setattr(planning, "minimize", fill_then_return)
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.1
    answer = planning.solve_prefix_preserving_trajectory_response(**arguments, max_evaluations=13)
    checked = audit(answer, classify(answer, arguments))
    phase = answer["phases"][0]
    assert phase["optimizer_returned"] and phase["status"] == "response_budget"
    assert phase["callback_count"] == 1 and phase["iterations"] == phase["optimizer_iterations"] == 3
    assert checked["termination_reason"] == "response_budget" and not checked["witness_indices"]
    assert not any(np.array_equal(phase["final_commands"], row["commands"]) for row in answer["evaluated_candidates"])


def test_last_response_witness_survives_prefix_interrupt_and_full_budget(monkeypatch, audit):
    def last_probe_wins(objective, start, *, callback, **_kwargs):
        for number in range(1, 12):
            probe = start.copy()
            probe[7] = number * 1e-5
            objective(probe)
        final = start.copy()
        final[[0, 6]] = [0.009, 0.258]
        objective(final)
        # A cached probe can win while the callback reports a different point.
        callback(start)
        pytest.fail("The completed witness must interrupt the optimizer")

    monkeypatch.setattr(planning, "minimize", last_probe_wins)
    arguments = inputs()
    arguments["desired"][:, 0, 3] = [0, 0.275]
    answer = planning.solve_prefix_preserving_trajectory_response(**arguments, max_evaluations=13)
    checked = audit(answer, classify(answer, arguments))
    assert answer["response_evaluations"] == 13
    assert checked["selected_trace_index"] == 12 and checked["termination_reason"] == "witness"
    phase = answer["phases"][0]
    assert phase["status"] == "prefix_advanced" and phase["iterations"] == 1
    assert not np.array_equal(phase["last_callback_commands"], phase["selected_commands"])


def test_disconnected_good_points_and_success_cannot_claim_growth(monkeypatch, audit):
    monkeypatch.setattr(planning, "minimize", lambda _objective, start, **_kwargs: optimizer_result(start, nit=2))
    arguments = inputs(horizon=3)
    arguments["desired"][:, 0, 3] = [0, 0.02, 0]
    answer = planning.solve_prefix_preserving_trajectory_response(**arguments, max_iterations=2)
    classifications = classify(answer, arguments)
    checked = audit(answer, classifications)
    assert checked["selected_prefix_length"] == 1 and not checked["witness_indices"]
    assert checked["termination_reason"] == "no_prefix_progress"
    assert len(checked["phase_checks"]) == 1
    # Counting all good points would invent progress across a failed cursor.
    corrupt = deepcopy(answer)
    corrupt["evaluated_candidates"][0]["prefix_length"] = 2
    with pytest.raises(ValueError, match="Strict archived and independent"):
        audit(corrupt, classifications)


def test_counted_callbacks_stop_a_growing_prefix_at_iteration_budget(monkeypatch, audit):
    def advance_on_last_iteration(objective, start, *, callback, **_kwargs):
        callback(start)
        partial = start.copy()
        partial[6] = 0.1
        objective(partial)
        callback(partial)
        pytest.fail("The second counted callback must end this phase")

    monkeypatch.setattr(planning, "minimize", advance_on_last_iteration)
    arguments = inputs(horizon=3)
    arguments["desired"][:, 0, 3] = [0, 0.1, 0.2]
    answer = planning.solve_prefix_preserving_trajectory_response(**arguments, max_iterations=2)
    checked = audit(answer, classify(answer, arguments))
    assert checked["total_iterations"] == 2 and checked["selected_prefix_length"] == 2
    assert checked["termination_reason"] == "iteration_budget"
    assert len(answer["phases"]) == 1 and not answer["witness_found"]


@pytest.mark.parametrize(
    "corruption",
    ["prefix", "rank", "phase_start", "response_boundary", "iteration_total", "phase_iterations", "termination"],
)
def test_archive_tampering_is_rejected(audit, two_phase_archive, corruption):
    original, classifications = two_phase_archive
    audit(original, classifications)
    answer = deepcopy(original)
    if corruption == "prefix":
        answer["evaluated_candidates"][1]["prefix_length"] = 3
    elif corruption == "rank":
        answer["selected"]["commands"] = answer["evaluated_candidates"][1]["commands"].copy()
    elif corruption == "phase_start":
        answer["phases"][1]["initial_commands"] = answer["used_start"].copy()
    elif corruption == "response_boundary":
        answer["phases"][1]["evaluations_before"] = 1
    elif corruption == "iteration_total":
        answer["total_iterations"] += 1
    elif corruption == "phase_iterations":
        answer["phases"][0]["iterations"] = 1
        answer["total_iterations"] -= 1
    elif corruption == "termination":
        answer["termination_reason"] = "no_prefix_progress"
    with pytest.raises(ValueError):
        audit(answer, classifications)


@pytest.mark.parametrize("archived_inside", [True, False])
def test_sub_tolerance_pose_boundary_disagreement_still_fails(monkeypatch, audit, archived_inside):
    monkeypatch.setattr(planning, "minimize", lambda _objective, start, **_kwargs: optimizer_result(start))
    arguments = inputs()
    arguments["desired"][1, 0, 3] = 0.01 if archived_inside else np.nextafter(0.01, np.inf)
    answer = planning.solve_prefix_preserving_trajectory_response(**arguments)
    classifications = classify(answer, arguments)
    audit(answer, classifications)
    independent = classifications[0]
    saved_norm = independent["normalized_squared_errors"][1, 0]
    # Simulate already reconstructed values on the opposite side of the exact
    # cap; a numerically tiny discrepancy still loses independent qualification.
    independent["normalized_squared_errors"][1, 0] = np.nextafter(1.0, np.inf) if archived_inside else 1.0
    independent["prefix_length"] = 1 if archived_inside else 2
    independent["feasible"] = not archived_inside
    assert abs(independent["normalized_squared_errors"][1, 0] - saved_norm) < 1e-10
    with pytest.raises(ValueError, match="Strict archived and independent"):
        audit(answer, classifications)


def test_complete_warm_inside_optimization_buffer_needs_no_phase(monkeypatch, audit):
    def terminal_tolerance_response(commands):
        result = response(commands)
        result["joint_velocity"][-1, 6] = 5e-11
        return result

    def no_optimizer(*_args, **_kwargs):
        pytest.fail("A complete audited warm must return without optimization")

    monkeypatch.setattr(planning, "minimize", no_optimizer)
    arguments = inputs(terminal_tolerance_response)
    arguments["terminal_normals"] = np.eye(7)[6:]
    answer = planning.solve_prefix_preserving_trajectory_response(**arguments)
    checked = audit(answer, classify(answer, arguments))
    assert checked["witness_indices"] == [0]
    assert checked["total_iterations"] == 0 and checked["phase_checks"] == []
