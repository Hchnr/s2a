"""The post-closure switch must preserve history and absolute reference time."""

import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import run_causal_event_switch_feedback as switch
from tests.research.joint_control.preview.test_loaded_gripper_feedback import switch_fixture


def test_real_input_and_switched_targets_use_the_two_previously_verified_chunks():
    inputs = switch.load_inputs()
    assert inputs["case"]["spec"]["case_id"] == "task1_init0_s42_terminal058"
    old, latest, poses, openings = switch.switched_targets(inputs["case"])
    np.testing.assert_array_equal(poses[:8], old["fixed_targets"][:8])
    np.testing.assert_array_equal(poses[8:], latest["fixed_targets"][8:])
    np.testing.assert_array_equal(openings[:8], old["fixed_openings"][:8])
    np.testing.assert_array_equal(openings[8:], latest["fixed_openings"][8:])


def test_switch_records_the_unchanged_completed_control_history():
    policy, arm, _refiner, _inner, _gripper = switch_fixture()
    arm.coupled.innovation = np.arange(12, dtype=float)
    arm.tracking.history.innovation = np.arange(12, dtype=float) / 10
    latest = {"targets": np.tile(np.eye(4), (40, 1, 1)), "openings": np.ones(40), "anchor": np.eye(4)}
    receipt = {}
    switch.switch_reference(policy, latest, receipt)
    assert receipt["reference_switch"]["completed"]
    assert receipt["reference_switch"]["detail"]["next_cursor"] == 8
    assert receipt["reference_switch"]["unchanged_history"]["arm_completed_periods"] == 368


def test_switch_refuses_to_hide_history_mutation():
    policy, arm, _refiner, _inner, _gripper = switch_fixture()
    arm.coupled.innovation = np.zeros(12)
    arm.tracking.history.innovation = np.zeros(12)

    def corrupt(*_args, **_kwargs):
        arm.previous[0] = 999
        return {}

    policy.rebind_reference = corrupt
    latest = {"targets": np.tile(np.eye(4), (40, 1, 1)), "openings": np.ones(40), "anchor": np.eye(4)}
    receipt = {}
    with pytest.raises(AssertionError):
        switch.switch_reference(policy, latest, receipt)
    assert receipt["reference_switch"]["attempted"] and not receipt["reference_switch"]["completed"]


def test_switch_checker_entry_does_not_import_candidate_policies():
    program = (
        "import sys; from scripts.simulation.runtime import run_causal_event_switch_feedback as s; "
        "s.base.checker_action([]); "
        "assert all(name not in sys.modules for name in s.base.FORBIDDEN_CHECKER_MODULES)"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
