"""Frozen deferred candidates cannot be replaced, clipped or silently resampled."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from scripts.tracking.trajectory_search.probe_deferred_cap_directions import select_cases


def fixture():
    spec = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "configs/tracking/trajectory_search/deferred_cap_directions_20261102.json"
        ).read_text()
    )
    x = np.zeros((12, 6))
    stages = [None] * 12
    for index, step, radius in [(9, 0.0006, 0.000625), (10, 0.0003, 0.0003125), (11, 0.00015, 0.00015625)]:
        point = np.zeros(73)
        point[0] = step
        command = x.copy()
        command[0, 0] = step
        t = dict(
            alpha=1.0,
            response_index=None,
            acceptance_reason="predicted_reduction_insufficient",
            predicted_reduction=8e-5,
            admission=dict(box=True, rate=True, global_box=True, trust=True, nonzero=True, unique=True),
            commands=command,
        )
        stages[index] = dict(
            mode="pose",
            qp=dict(qualified=True, point=point),
            anchor_before_index=18,
            radius_before=radius,
            attempts=[t],
        )
    answer = dict(
        evaluated_candidates=[dict(commands=x.copy()) for _ in range(19)],
        stages=stages,
        global_lower=np.full((12, 6), -0.1),
        global_upper=np.full((12, 6), 0.1),
    )
    return answer, spec, np.zeros(6), -np.ones(6), np.ones(6)


def test_all_three_frozen_directions_and_repeat_selected_without_input_mutation():
    args = fixture()
    before = deepcopy(args[0])
    anchor, cases = select_cases(*args)
    assert len(cases) == 5 and [c["stage_index"] for c in cases[1:4]] == [9, 10, 11]
    assert_array_equal(cases[0]["commands"], cases[-1]["commands"])
    for case in cases[1:4]:
        assert_array_equal(case["commands"], before["stages"][case["stage_index"]]["attempts"][0]["commands"])
    assert_array_equal(anchor["commands"], before["evaluated_candidates"][18]["commands"])


@pytest.mark.parametrize(
    "fault",
    [
        "already_called",
        "wrong_reason",
        "nonpositive",
        "large_prediction",
        "unqualified",
        "partial",
        "global",
        "duplicate",
        "budget",
    ],
)
def test_invalid_fixed_case_aborts_instead_of_replacement(fault):
    answer, spec, previous, lo, hi = fixture()
    stage = answer["stages"][9]
    t = stage["attempts"][0]
    if fault == "already_called":
        t["response_index"] = 1
    elif fault == "wrong_reason":
        t["acceptance_reason"] = "actual_admission_failed"
    elif fault == "nonpositive":
        t["predicted_reduction"] = 0
    elif fault == "large_prediction":
        t["predicted_reduction"] = 1e-4
    elif fault == "unqualified":
        stage["qp"]["qualified"] = False
    elif fault == "partial":
        t["commands"] = np.zeros((11, 6))
    elif fault == "global":
        answer["global_upper"][0, 0] = 0.0001
    elif fault == "duplicate":
        other = answer["stages"][10]
        other["attempts"][0]["commands"] = t["commands"].copy()
        other["qp"]["point"] = stage["qp"]["point"].copy()
        other["radius_before"] = stage["radius_before"]
    else:
        spec["total_response_budget"] += 2
    with pytest.raises(ValueError):
        select_cases(answer, spec, previous, lo, hi)
