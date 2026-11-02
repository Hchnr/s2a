"""Reserve-aware pose acceptance and inherited root geometry remain explicit."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective
from s2a.research.trajectory_search.objective_search_driver import search
from s2a.research.trajectory_search.reserve_entry_acceptance import InheritedSearchDomain, ReserveEntryAcceptance
from scripts.tracking.trajectory_search.cap_distance_search_audit import audit_cap_search
from tests.research.trajectory_search.test_cap_distance_audit import archive
from tests.research.trajectory_search.test_objective_search_driver import solver
from tests.research.trajectory_search.test_state_vector_gn_search import entry, inputs
from tests.research.trajectory_search.test_state_vector_gn_search_audit import vector_inputs


def reserve_case():
    value = inputs(max_responses=6)

    def response(command):
        step = command[0, 0]
        E = np.zeros(72)
        E[54] = 1.5 - 10 * step
        reserve = 1.2 if step == 0 else (0.8 if step >= 0.0025 else 0.95)
        return entry(command, E, reserve=reserve)

    value["evaluate"] = response
    value["historical_initial"] = response(np.zeros((12, 6)))
    return value


def test_new_policy_rejects_larger_reserve_consuming_steps_then_accepts_smaller():
    value = reserve_case()
    answer = search(
        **value, objective=CapDistanceObjective(), qp_solver=solver(0.005), pose_acceptance=ReserveEntryAcceptance()
    )
    attempts = answer["stages"][0]["attempts"]
    assert [a["alpha"] for a in attempts] == [1.0, 0.5, 0.25]
    assert [a["acceptance_reason"] for a in attempts] == ["pose_reserve_below_entry"] * 2 + ["accepted_reduction"]
    assert answer["response_evaluations"] == 5
    assert answer["repeatability"]["exact"] and not answer["qualified_witness_indices"]
    assert answer["stages"][0]["rejected_update"] is None


def test_default_still_accepts_first_original_eligible_step_without_new_metadata():
    answer = search(**reserve_case(), objective=CapDistanceObjective(), qp_solver=solver(0.005))
    assert len(answer["stages"][0]["attempts"]) == 1
    assert answer["stages"][0]["accepted_response_index"] == 1
    assert "pose_acceptance_policy" not in answer and "inherited_domain" not in answer


def test_restore_mode_does_not_require_pose_reserve_gate():
    assert ReserveEntryAcceptance.allows(dict(reserve=0.1), dict(reserve_entry_fraction=0.9), "restore")
    assert not ReserveEntryAcceptance.allows(dict(reserve=0.899999), dict(reserve_entry_fraction=0.9), "pose")
    assert ReserveEntryAcceptance.allows(dict(reserve=0.9), dict(reserve_entry_fraction=0.9), "pose")


def test_full_original_buffered_witness_is_not_redefined_by_internal_reserve():
    value = reserve_case()
    old = value["evaluate"]

    def response(command):
        if not np.any(command):
            return old(command)
        return entry(command, np.zeros(72), reserve=0.8)

    value["evaluate"] = response
    answer = search(
        **value, objective=CapDistanceObjective(), qp_solver=solver(0.005), pose_acceptance=ReserveEntryAcceptance()
    )
    assert answer["qualified_witness_indices"] == [1]
    assert answer["termination_reason"] == "qualified_witness"


def test_root_geometry_is_inherited_without_recentering_or_new_pose_ceiling():
    value = reserve_case()
    root = deepcopy(value["historical_initial"])
    domain = InheritedSearchDomain.from_entry(root)
    shifted = np.zeros((12, 6))
    shifted[0, 0] = 0.09
    value["historical_initial"] = value["evaluate"](shifted)
    answer = search(**value, objective=CapDistanceObjective(), qp_solver=solver(0, qualified=False), domain=domain)
    assert_array_equal(answer["global_lower"], -np.ones((12, 6)) * 0.1)
    assert_array_equal(answer["global_upper"], np.ones((12, 6)) * 0.1)
    assert_array_equal(answer["pose_ceiling"], np.full(24, 2.25 * 1.01))


def test_current_anchor_outside_original_root_domain_is_rejected():
    value = reserve_case()
    domain = InheritedSearchDomain.from_entry(value["historical_initial"])
    shifted = np.zeros((12, 6))
    shifted[0, 0] = 0.15
    value["historical_initial"] = value["evaluate"](shifted)
    with pytest.raises(ValueError, match="initial command admission"):
        search(**value, objective=CapDistanceObjective(), domain=domain)


def test_default_independent_audit_rejects_unregistered_variant_metadata():
    evaluate, args, _ = vector_inputs(curvature=20.0)
    answer = archive(evaluate, args)
    answer["pose_acceptance_policy"] = ReserveEntryAcceptance.policy
    with pytest.raises(ValueError, match="unaudited acceptance"):
        audit_cap_search(answer, *args)
