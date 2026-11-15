"""Matched-state comparisons must include controller and motor cache fields."""

import copy

import numpy as np
import pytest

from s2a.research.native_runtime.development_prefix_replay import SAMPLE_FIELDS
from scripts.simulation.runtime.run_correct_reference_same_state_arm import compare_branchpoints


def test_branchpoint_comparison_rejects_changed_controller_or_motor_cache():
    sample = {name: np.asarray([1.0, 2.0]) for name in (*SAMPLE_FIELDS, "gripper_current_action")}
    first = dict(sample=sample, endpoint_receipt=dict(endpoint_step=40, sim_state_sha256="original"))
    second = copy.deepcopy(first)
    assert compare_branchpoints(first, second) == 0
    second["sample"]["controller_goal_pos"][0] += 0.001
    with pytest.raises(ValueError, match="controller_goal_pos"):
        compare_branchpoints(first, second)
    second = copy.deepcopy(first)
    second["sample"]["gripper_current_action"][0] += 0.001
    with pytest.raises(ValueError, match="gripper_current_action"):
        compare_branchpoints(first, second)
