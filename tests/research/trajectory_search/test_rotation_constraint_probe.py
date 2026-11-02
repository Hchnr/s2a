"""One-group objective selection retains every constraint and preflight gate."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective
from scripts.tracking.trajectory_search.probe_rotation_constraints import preflight, rotation_problem, validate_spec
from tests.research.trajectory_search.test_state_vector_gn_search import entry, inputs


def test_rotation_selection_removes_other_objectives_and_preserves_all_hard_rows():
    vector = np.zeros(72)
    vector[0] = vector[69] = 2
    anchor = entry(np.zeros((12, 6)), vector)
    models = dict(Js=np.zeros((4369, 72)), Jq=np.zeros((24, 72)), Je=np.zeros((72, 72)))
    models["Je"][0, 0] = 5
    models["Je"][69, 66] = -10
    previous = np.zeros(6)
    lo, hi = -np.ones((12, 6)) * 0.1, np.ones((12, 6)) * 0.1
    ceiling = np.full(24, 5.0)
    config = inputs()["config"]
    problem = rotation_problem(anchor, models, previous, lo, hi, ceiling, 0.005, config)
    full = CapDistanceObjective.problem(
        anchor, models["Js"], models["Jq"], models["Je"], previous, lo, hi, ceiling, 0.005, "pose", config
    )
    for field in ["A", "b", "initial_t"]:
        assert_array_equal(problem[field], full[field])
    assert problem["B"][0, 0] == 1 and full["B"][0, 0] == 26
    assert problem["B"][66, 66] == 101 and problem["g"][66] == -10
    assert problem["g"][0] == 0 and full["g"][0] == 5


def test_preflight_does_not_clip_or_admit_duplicates_and_oversized_steps():
    anchor = np.zeros((12, 6))
    command = anchor.copy()
    command[-1, 0] = 0.001
    args = (anchor, np.zeros(6), -np.ones(6), np.ones(6), -np.ones((12, 6)) * 0.1, np.ones((12, 6)) * 0.1, 0.005)
    assert all(preflight(command, *args, set()).values())
    assert not preflight(command, *args, {command.tobytes()})["unique"]
    command[-1, 0] = 0.2
    original = command.copy()
    gates = preflight(command, *args, set())
    assert not gates["trust"] and not gates["global_box"]
    assert_array_equal(command, original)


@pytest.mark.parametrize("change", [None, "target", "budget", "radius", "retry"])
def test_fixed_probe_specification(change):
    path = (
        Path(__file__).resolve().parents[3]
        / "configs/tracking/trajectory_search/rotation_constraint_probe_20261102.json"
    )
    spec = json.loads(path.read_bytes())
    if change is None:
        validate_spec(spec)
        return
    if change == "target":
        spec["target_group"] = 22
    elif change == "budget":
        spec["total_response_budget"] += 1
    elif change == "radius":
        spec["radius"] *= 2
    else:
        spec["automatic_retry"] = True
    with pytest.raises(ValueError):
        validate_spec(spec)
