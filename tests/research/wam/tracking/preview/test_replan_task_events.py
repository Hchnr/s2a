"""Opening events compare absolute endpoints, without future observations."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.wam.tracking.preview.replan_task_events import compare_replans


def inputs():
    return {
        "old_poses": np.tile(np.eye(4), (40, 1, 1)),
        "old_openings": np.ones(40),
        "new_poses": np.tile(np.eye(4), (40, 1, 1)),
        "new_openings": np.ones(40),
        "old_step": 8,
        "new_step": 16,
        "measured_tcp": np.eye(4),
        "opening": 0.9,
    }


def test_matching_deferred_closure_uses_same_absolute_endpoint_and_does_not_mutate_inputs():
    x = inputs()
    x["old_openings"][20:] = 0.1
    x["new_openings"][12:] = 0.1
    before = copy.deepcopy(x)
    result = compare_replans(**x)
    assert result["closure"]["deferred_in_old"]
    assert result["closure"]["old_absolute_endpoint"] == result["closure"]["new_first_absolute_endpoint"] == 29
    assert result["closure"]["first_event_revision"] == "same"
    assert result["closure"]["currently_opposite_phase"] and result["closure"]["event_pose_agreement"]
    assert result["closure"]["next_prefix_pose_agreement"] and result["closure"]["current_near_old_prefix"]
    assert result["old_closed_new_open"] == result["old_open_new_closed"] == 0
    for key in ("old_poses", "new_poses", "old_openings", "new_openings", "measured_tcp"):
        np.testing.assert_array_equal(x[key], before[key])


@pytest.mark.parametrize(
    "new_first,category,shift", [(None, "not_in_new_horizon", None), (3, "earlier", -9), (22, "later", 10)]
)
def test_first_event_revision_covers_withdrawal_earlier_and_later(new_first, category, shift):
    x = inputs()
    x["old_openings"][20:] = 0.1
    if new_first is not None:
        x["new_openings"][new_first:] = 0.1
    event = compare_replans(**x)["closure"]
    assert event["first_event_revision"] == category and event["new_first_shift_steps"] == shift
    assert event["opposite_opening_at_old_event"] is (new_first is None or new_first > 12)


def test_first_event_already_inside_executed_prefix_is_not_deferred():
    x = inputs()
    x["old_openings"][7:] = 0.1
    event = compare_replans(**x)["closure"]
    assert event == {"old_first_index": 7, "new_first_index": None, "deferred_in_old": False}


def test_opening_event_and_measured_opposite_phase_are_reported_separately():
    x = inputs()
    x["old_openings"][:] = 0.1
    x["old_openings"][8:] = 0.9
    x["new_openings"][:] = 0.1
    x["opening"] = 0.2
    result = compare_replans(**x)
    event = result["opening"]
    assert event["old_absolute_endpoint"] == 17 and event["first_event_revision"] == "not_in_new_horizon"
    assert event["currently_opposite_phase"] and event["opposite_opening_at_old_event"]
    assert result["old_open_new_closed"] == 32


def test_full_prefix_and_event_pose_gates_are_distinct_and_use_so3():
    x = inputs()
    x["old_openings"][20:] = 0.1
    x["new_poses"][1, :3, :3] = Rotation.from_rotvec([0, 0, 0.11]).as_matrix()
    x["new_poses"][12, 0, 3] = 0.021
    result = compare_replans(**x)
    assert result["rotation_difference_rad"][1] == pytest.approx(0.11)
    assert not result["next_prefix_pose_agreement"]
    assert not result["closure"]["event_pose_agreement"]
    assert result["current_near_old_prefix"]


@pytest.mark.parametrize("invalid", ["step", "shape", "opening", "pose", "current"])
def test_invalid_time_or_reference_is_rejected(invalid):
    x = inputs()
    if invalid == "step":
        x["new_step"] += 1
    elif invalid == "shape":
        x["new_openings"] = x["new_openings"][:-1]
    elif invalid == "opening":
        x["new_openings"][0] = 1.01
    elif invalid == "pose":
        x["old_poses"][0, 0, 0] = 2
    else:
        x["opening"] = float("nan")
    with pytest.raises(ValueError):
        compare_replans(**x)
