"""Near-cap units, unsafe witness rejection and finite search state transitions."""

import numpy as np

from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective
from s2a.research.trajectory_search.objective_search_driver import search
from tests.research.trajectory_search.test_state_vector_gn_search import entry, inputs


def solver(step, *, qualified=True):
    def solve(*args, **kwargs):
        point = np.zeros(73)
        point[0] = step
        return dict(qualified=qualified, point=point, solver_nit=0)

    return solve


def near_cap():
    value = inputs(max_responses=3)
    E = np.zeros(72)
    E[54] = 1.0001
    derivative = np.zeros(72)
    derivative[54] = -10
    value["evaluate"] = lambda command: entry(command, E + derivative * command[0, 0])
    value["historical_initial"] = value["evaluate"](np.zeros((12, 6)))
    value["Jq"][18, 0] = -20.002
    return value


def test_near_cap_progress_is_not_blocked_by_old_pose_excess_units():
    value = near_cap()
    answer = search(**value, objective=CapDistanceObjective(), qp_solver=solver(5e-6))
    trial = answer["stages"][0]["attempts"][0]
    assert trial["actual_reduction"] < 1e-8
    assert trial["accepted"] and trial["acceptance_reason"] == "accepted_reduction"
    assert answer["witness_indices"] == []
    assert [r["role"] for r in answer["request_ledger"]] == ["initial", "trial", "initial_repeat"]
    assert answer["repeatability"]["exact"]


def test_full_original_witness_remains_the_only_success_gate():
    answer = search(**near_cap(), objective=CapDistanceObjective(), qp_solver=solver(1.1e-5))
    assert answer["qualified_witness_indices"] == [1]
    assert answer["termination_reason"] == "qualified_witness"
    assert answer["control_qualified"] is False


def test_inside_caps_with_failed_joint_audit_is_rejected_and_still_charged():
    value = near_cap()
    original = value["evaluate"]

    def response(command):
        observed = original(command)
        if np.any(command):
            observed["audit"]["feasible"] = False
            observed["feasible"] = False
        return observed

    value["evaluate"] = response
    answer = search(**value, objective=CapDistanceObjective(), qp_solver=solver(1.1e-5))
    assert not answer["witness_indices"]
    assert answer["stages"][0]["attempts"][0]["acceptance_reason"] == "actual_admission_failed"
    assert answer["response_evaluations"] == 3
    assert answer["repeatability"]["exact"]


def test_unqualified_qps_shrink_and_stop_without_extra_responses():
    value = inputs()
    value["config"]["max_stages"] = 60
    answer = search(**value, objective=CapDistanceObjective(), qp_solver=solver(0, qualified=False))
    assert answer["termination_reason"] == "radius_below_minimum"
    assert len(answer["stages"]) == 6
    assert answer["response_evaluations"] == 2
    assert not any(stage["attempts"] for stage in answer["stages"])
