"""Aligned reserve floors tighten safety only, with independent reconstruction."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective
from s2a.research.trajectory_search.objective_search_driver import search
from s2a.research.trajectory_search.reserve_aligned_cap_objective import ReserveAlignedCapObjective
from scripts.tracking.trajectory_search.cap_distance_search_audit import audit_cap_search
from scripts.tracking.trajectory_search.reserve_aligned_cap_audit import ReserveAlignedCapAuditPolicy
from tests.research.trajectory_search.test_state_vector_dual_recovery_audit import annotate
from tests.research.trajectory_search.test_state_vector_gn_search import inputs
from tests.research.trajectory_search.test_state_vector_gn_search_audit import hand_certificate, vector_inputs


def problem(policy, value, mode):
    e = value["historical_initial"]
    q = e["normalized_squared_errors"].ravel()
    cfg = value["config"]
    return policy.problem(
        e,
        value["Js"],
        value["Jq"],
        value["Je"],
        value["previous"],
        np.full((12, 6), -0.1),
        np.full((12, 6), 0.1),
        np.full(24, max(1, float(q.max())) * (1 + cfg["pose_ceiling_fraction"])),
        0.005,
        mode,
        cfg,
    )


def independent(value, mode):
    e = value["historical_initial"]
    q = e["normalized_squared_errors"].ravel()
    cfg = value["config"]
    error = (e["tcp_errors"] / np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])).ravel()
    return ReserveAlignedCapAuditPolicy.problem(
        e["commands"],
        e["safety_slack"],
        q,
        error,
        value["Js"],
        value["Jq"],
        value["Je"],
        value["previous"],
        np.full((12, 6), -0.1),
        np.full((12, 6), 0.1),
        np.full(24, max(1, float(q.max())) * (1 + cfg["pose_ceiling_fraction"])),
        0.005,
        mode,
        cfg,
    )


@pytest.mark.parametrize("mode", ["pose", "restore"])
def test_objective_and_other_rows_unchanged_and_independent_arrays_exact(mode):
    value = inputs()
    old = problem(CapDistanceObjective, value, mode)
    new = problem(ReserveAlignedCapObjective, value, mode)
    for k in ["A", "B", "g", "command_lower", "command_upper"]:
        assert_array_equal(old[k], new[k])
    assert_array_equal(old["b"][4369:], new["b"][4369:])
    if mode == "pose":
        assert np.all(new["b"][:4369] <= old["b"][:4369]) and np.any(new["b"][:4369] < old["b"][:4369])
    else:
        assert_array_equal(old["b"], new["b"])
    for k, v in zip(["B", "g", "A", "b"], independent(value, mode), strict=True):
        assert_array_equal(new[k], v)


def test_terminal_time_conversion_and_zero_pose_step_feasibility():
    value = inputs()
    cfg = value["config"]
    p = problem(ReserveAlignedCapObjective, value, "pose")
    assert p["safety_floor"][0] == cfg["safety_buffer"] + 0.9 * cfg["reserve_target"]
    assert p["safety_floor"][-1] == cfg["safety_buffer"] + 0.9 * cfg["reserve_target"] / cfg["terminal_time_scale"]
    assert np.all(p["b"][:4369] >= 0)


def test_alignment_never_lowers_a_stricter_original_floor():
    value = inputs()
    value["config"]["reserve_entry_fraction"] = 0.2
    old = problem(CapDistanceObjective, value, "pose")
    new = problem(ReserveAlignedCapObjective, value, "pose")
    assert_array_equal(new["safety_floor"], old["safety_floor"])
    assert_array_equal(new["b"], old["b"])


def test_independent_policy_never_calls_production_floor(monkeypatch):
    value = inputs()
    expected = independent(value, "pose")

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent floor cannot call production")

    monkeypatch.setattr(ReserveAlignedCapObjective, "problem", forbidden)
    for x, y in zip(expected, independent(value, "pose"), strict=True):
        assert_array_equal(x, y)


def aligned_archive():
    evaluate, args, _ = vector_inputs(curvature=20.0)
    Js, Jq, Je, initial, previous, lo, hi, cfg = args

    def solve(B, g, A, b, initial_t, **kwargs):
        qp = hand_certificate(B, g, A, b, initial_t, **kwargs)
        return annotate(qp, A, b, kwargs["dual_recovery_mode"], kwargs["objective_divisor"])

    answer = search(
        evaluate, initial, Js, Jq, Je, previous, lo, hi, cfg, objective=ReserveAlignedCapObjective(), qp_solver=solve
    )
    return answer, args


def test_full_ledger_requires_explicit_aligned_policy_and_rejects_corrupt_safety_certificate():
    answer, args = aligned_archive()
    assert audit_cap_search(answer, *args, objective_audit=ReserveAlignedCapAuditPolicy())["passed"]
    with pytest.raises(ValueError):
        audit_cap_search(answer, *args)
    bad = deepcopy(answer)
    bad["stages"][0]["qp"]["row_scale"][0] += 1e-6
    with pytest.raises(ValueError):
        audit_cap_search(bad, *args, objective_audit=ReserveAlignedCapAuditPolicy())
