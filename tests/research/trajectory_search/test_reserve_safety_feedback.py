"""Safe charged source selection, old-feedback priority and Js-only state updates."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective
from s2a.research.trajectory_search.objective_search_driver import search
from s2a.research.trajectory_search.reserve_entry_acceptance import ReserveEntryAcceptance
from s2a.research.trajectory_search.reserve_safety_feedback import ReserveSafetyFeedback
from tests.research.trajectory_search.test_state_vector_gn_search import entry, inputs


def selection_case():
    value = inputs()
    cfg = value["config"]
    x = np.zeros((12, 6))
    E = np.zeros(72)
    E[54] = 1.5
    records = [entry(x, E, reserve=1.2), entry(x, E, reserve=0.8), entry(x, E, reserve=0.7)]
    step = x.copy()
    step[0, 0] = 0.001
    attempts = [
        dict(
            response_index=i,
            accepted=False,
            acceptance_reason="pose_reserve_below_entry",
            predicted_metrics=dict(reserve=1.0),
            actual_step=step.copy(),
            alpha=alpha,
        )
        for i, alpha in [(1, 0.5), (2, 0.25)]
    ]
    return dict(mode="pose", accepted_response_index=None, attempts=attempts), records, np.full(24, 3.0), cfg


def test_minimum_alpha_then_index_is_chosen_without_mutating_records():
    stage, records, ceiling, cfg = selection_case()
    assert ReserveSafetyFeedback.choose(stage, records, ceiling, cfg)["response_index"] == 2
    stage["attempts"][0]["alpha"] = 0.25
    assert ReserveSafetyFeedback.choose(stage, records, ceiling, cfg)["response_index"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "accepted_stage",
        "restore",
        "unbilled",
        "accepted_trial",
        "wrong_reason",
        "predicted_below",
        "predicted_nan",
        "unsafe",
        "buffer",
        "ceiling",
        "zero",
        "small",
        "actual_above",
    ],
)
def test_source_eligibility_rejects_each_unqualified_case(fault):
    stage, records, ceiling, cfg = selection_case()
    stage["attempts"] = stage["attempts"][1:]
    t = stage["attempts"][0]
    e = records[2]
    if fault == "accepted_stage":
        stage["accepted_response_index"] = 1
    elif fault == "restore":
        stage["mode"] = "restore"
    elif fault == "unbilled":
        t["response_index"] = None
    elif fault == "accepted_trial":
        t["accepted"] = True
    elif fault == "wrong_reason":
        t["acceptance_reason"] = "actual_reduction_insufficient"
    elif fault == "predicted_below":
        t["predicted_metrics"]["reserve"] = np.nextafter(0.9, 0)
    elif fault == "predicted_nan":
        t["predicted_metrics"]["reserve"] = np.nan
    elif fault == "unsafe":
        e["audit"]["feasible"] = False
    elif fault == "buffer":
        e["safety_slack"][0] = 0
    elif fault == "ceiling":
        ceiling[:] = 1
    elif fault == "zero":
        t["actual_step"][:] = 0
    elif fault == "small":
        t["actual_step"][:] = 0
        t["actual_step"][0, 0] = 1e-8
    else:
        e["safety_slack"][:] = 0.001
    assert ReserveSafetyFeedback.choose(stage, records, ceiling, cfg) is None


def search_case(*, pose_ceiling_failure=False):
    value = inputs(max_responses=12)
    value["config"]["max_stages"] = 2

    def response(command):
        E = np.zeros(72)
        E[54] = 1.5 + (100 if pose_ceiling_failure else -10) * command[0, 0]
        return entry(command, E, reserve=1.2 if not np.any(command) else 0.8)

    value["evaluate"] = response
    value["historical_initial"] = response(np.zeros((12, 6)))
    calls = []

    def solve(B, g, A, b, initial_t, **kwargs):
        calls.append(dict(B=B.copy(), g=g.copy(), A=A.copy()))
        point = np.zeros(73)
        point[0] = 0.005
        return dict(qualified=len(calls) == 1, point=point, solver_nit=0)

    return value, solve, calls


def test_js_only_feedback_preserves_anchor_pose_models_and_actual_call_budget():
    value, solve, calls = search_case()
    before = deepcopy(value)
    answer = search(
        **value,
        objective=CapDistanceObjective(),
        qp_solver=solve,
        pose_acceptance=ReserveEntryAcceptance(),
        reserve_feedback=ReserveSafetyFeedback(),
    )
    update = answer["stages"][0]["rejected_update"]
    assert update["kind"] == "reserve_safety_only" and update["source_response_index"] == 5
    assert update["pose"] is update["error_vector"] is None
    assert answer["rejected_update_count"] == 1 and answer["response_evaluations"] == 7
    assert answer["stages"][0]["anchor_after_index"] == answer["stages"][1]["anchor_before_index"] == 0
    assert_array_equal(calls[0]["B"], calls[1]["B"])
    assert_array_equal(calls[0]["g"], calls[1]["g"])
    assert_array_equal(calls[0]["A"][4369:], calls[1]["A"][4369:])
    assert np.any(calls[0]["A"][:4369] != calls[1]["A"][:4369])
    for k in ["Js", "Jq", "Je"]:
        assert_array_equal(value[k], before[k])
    assert answer["repeatability"]["exact"] and not answer["qualified_witness_indices"]


def test_old_pose_ceiling_feedback_keeps_priority_and_three_model_update():
    value, solve, _ = search_case(pose_ceiling_failure=True)
    value["config"]["max_stages"] = 1

    class UnusedPolicy(ReserveSafetyFeedback):
        @staticmethod
        def choose(*args):
            raise AssertionError("Original feedback must take priority")

    answer = search(
        **value,
        objective=CapDistanceObjective(),
        qp_solver=solve,
        pose_acceptance=ReserveEntryAcceptance(),
        reserve_feedback=UnusedPolicy(),
    )
    update = answer["stages"][0]["rejected_update"]
    assert update is not None and "kind" not in update
    assert update["pose"] is not None and update["error_vector"] is not None


def test_default_three_model_feedback_still_passes_complete_independent_audit():
    from scripts.tracking.trajectory_search.cap_distance_search_audit import audit_cap_search
    from tests.research.trajectory_search.test_cap_distance_audit import archive
    from tests.research.trajectory_search.test_state_vector_gn_search_audit import restoration_inputs

    evaluate, arguments = restoration_inputs(feedback=True)
    answer = archive(evaluate, arguments)
    assert answer["rejected_update_count"] > 0
    assert "reserve_feedback_policy" not in answer
    assert audit_cap_search(answer, *arguments)["passed"]
