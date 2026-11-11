import copy

import numpy as np
import pytest

from s2a.research.joint_control.preview.ordered_correct_arm_history import restore_ordered_arm_history
from scripts.simulation.runtime.rebuild_ordered_correct_gripper import ROOT, load_inputs, read_checked


def recorded_inputs():
    inputs = load_inputs()
    config = inputs["config"]
    row = read_checked(ROOT / config["rows"], config["rows_sha256"])[80]
    controller = read_checked(inputs["refs"]["controller"]["path"], inputs["refs"]["controller"]["sha256"])
    return (
        inputs["rows"][-1],
        row["site_pose"],
        row["site_twist"],
        controller["contract"],
        row["refinement"]["history_before"],
    )


def test_real_completed_predictions_reconstruct_two_histories():
    previous, site, twist, contract, tracking_before = recorded_inputs()
    coupled, tracking, difference = restore_ordered_arm_history(previous, site, twist, contract, tracking_before)
    assert coupled.completed_periods == tracking.history.completed_periods == 80
    assert max(difference.values()) < 1e-8
    np.testing.assert_array_equal(tracking.history.last_plan, previous["refinement"]["planned_commands"])
    np.testing.assert_allclose(coupled.current(site, twist), previous["coupled_innovation_after"], atol=1e-8, rtol=0)


@pytest.mark.parametrize("corruption", ["prediction", "current_state", "tracking_innovation"])
def test_rejects_history_not_explained_by_completed_prediction(corruption):
    previous, site, twist, contract, tracking_before = recorded_inputs()
    previous, site, tracking_before = copy.deepcopy(previous), copy.deepcopy(site), copy.deepcopy(tracking_before)
    if corruption == "prediction":
        previous["refinement"]["nominal_response"][0] += 0.01
    elif corruption == "current_state":
        site[0][3] += 0.01
    else:
        tracking_before[0] += 0.01
    with pytest.raises(ValueError, match="Prior innovations"):
        restore_ordered_arm_history(previous, site, twist, contract, tracking_before)
