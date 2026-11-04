"""Preregistered support probes retain the saved direction, domain and budgets."""

import json
from copy import deepcopy

import numpy as np
import pytest

from scripts.tracking.trajectory_search.probe_local_support_direction import CONFIG, select_cases


def fixture():
    spec = json.loads(CONFIG.read_text())
    answer = dict(
        evaluated_candidates=[dict(commands=np.zeros((12, 6))) for _ in range(24)],
        global_lower=np.full((12, 6), -0.2),
        global_upper=np.full((12, 6), 0.2),
    )
    point = np.zeros(72)
    point[0] = 0.1
    lp = dict(success=True, anchor_index=23, group=23, reserve_fraction=0.9, point=point)
    return answer, lp, spec, np.zeros(6), -np.ones(6), np.ones(6)


def test_fixed_direction_scaling_repeat_and_input_immutability():
    args = fixture()
    original = deepcopy(args)
    anchor, cases = select_cases(*args)
    assert len(cases) == 5
    np.testing.assert_array_equal(cases[0]["commands"], cases[-1]["commands"])
    for i, alpha in enumerate([1.0, 0.5, 0.25], start=1):
        np.testing.assert_array_equal(
            cases[i]["commands"], np.asarray(anchor["commands"]) + alpha * original[1]["point"].reshape(12, 6)
        )
    np.testing.assert_array_equal(args[1]["point"], original[1]["point"])


@pytest.mark.parametrize(
    "fault",
    ["budget", "alphas", "group", "anchor", "reserve", "failed", "partial", "nan", "zero", "global", "input", "rate"],
)
def test_no_replacement_clipping_or_admission_bypass(fault):
    answer, lp, spec, prev, lo, hi = fixture()
    if fault == "budget":
        spec["total_response_budget"] += 2
    elif fault == "alphas":
        spec["alphas"] = [0.5, 0.25, 0.125]
    elif fault == "group":
        lp["group"] = 22
    elif fault == "anchor":
        lp["anchor_index"] = 0
    elif fault == "reserve":
        lp["reserve_fraction"] = 0
    elif fault == "failed":
        lp["success"] = False
    elif fault == "partial":
        lp["point"] = lp["point"][:-1]
    elif fault == "nan":
        lp["point"][0] = np.nan
    elif fault == "zero":
        lp["point"][:] = 0
    elif fault == "global":
        answer["global_upper"][0, 0] = 0.05
    elif fault == "input":
        hi[0] = 0.05
    else:
        prev[0] = -0.2
    with pytest.raises(ValueError):
        select_cases(answer, lp, spec, prev, lo, hi)
