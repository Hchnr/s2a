"""Independent release screening detects changed safety or timing evidence."""

from copy import deepcopy

import pytest

from scripts.verification.reference_ports import check_causal_opening_event_screen as checker


@pytest.fixture(scope="module")
def screen_cases():
    return checker.load_screen()[1]


def test_independent_reconstruction_covers_all_four_first_interventions(screen_cases):
    results = [checker.check_case(case) for case in screen_cases]
    assert [item["case"]["case_index"] for item in results] == [3, 7, 10, 11]
    assert all(not item["prior_inner_samples_checked"] and not item["physical_feedback_executed"] for item in results)


@pytest.mark.parametrize("fault", ["admission", "margin", "q", "event", "future"])
def test_independent_checker_refuses_tampered_admission(screen_cases, fault):
    data = deepcopy(screen_cases[0])
    if fault == "admission":
        data["admission"]["recorded_endpoint_admitted"] = not data["admission"]["recorded_endpoint_admitted"]
    elif fault == "margin":
        data["admission"]["minimum_margin_rad"] += 0.01
    elif fault == "q":
        data["admission"]["joint_positions_rad"][0] += 0.1
    elif fault == "event":
        data["case"]["event_endpoint"] += 1
    else:
        data["physical_feedback_executed"] = True
    with pytest.raises((AssertionError, ValueError)):
        checker.check_case(data)
