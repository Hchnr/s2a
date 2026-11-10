"""Long feedback must preserve the independently checked eight-period prefix."""

import copy
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_causal_event_feedback as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_long_feedback as long


@pytest.fixture(scope="module")
def inputs():
    return long.load_inputs()


def test_long_input_binds_only_the_two_verified_task1_branches(inputs):
    assert inputs["case"]["spec"]["case_id"] == "task1_init0_s42_terminal058"
    assert set(inputs["prior_branches"]) == {"latest", "remembered"}
    for mode in long.BRANCHES:
        selected = binding.selected_case(inputs["case"], mode, horizon=16)
        assert selected["fixed_targets"].shape == (16, 4, 4)
        assert len(selected["fixed_openings"]) == 16


@pytest.mark.parametrize("corruption", ["action", "state", "inner", "grasp", "task"])
def test_first_eight_period_comparison_rejects_altered_physical_evidence(inputs, corruption):
    old = prefix.read(inputs["prior_branches"]["remembered"]["producer_ref"])["runtime"]["continuation"]["periods"][0]
    changed = copy.deepcopy(old)
    if corruption == "action":
        changed["command"][6] += 0.1
    elif corruption == "state":
        changed["actual_state"][0] += 0.001
    elif corruption == "inner":
        changed["actual_inner"][0]["q"][0] += 0.001
    elif corruption == "grasp":
        changed["target_grasp"] = not changed["target_grasp"]
    else:
        changed["task_success"] = not changed["task_success"]
    with pytest.raises((ValueError, AssertionError)):
        long.check_previous_period(changed, old)


def test_target_position_is_read_only_and_validated():
    position = np.array([[0.1, 0.2, 0.3]])
    env = SimpleNamespace(
        env=SimpleNamespace(obj_body_id={"target": 0}), sim=SimpleNamespace(data=SimpleNamespace(body_xpos=position))
    )
    result = long.target_position(env, "target")
    result[0] = 9
    assert position[0, 0] == 0.1
    with pytest.raises(ValueError, match="evaluation object"):
        long.target_position(env, "missing")


def test_checker_entry_does_not_import_candidate_policies():
    program = (
        "import sys; from scripts.simulation.runtime import run_causal_event_long_feedback as r; "
        "r.previous.checker_action([]); "
        "assert all(name not in sys.modules for name in r.previous.FORBIDDEN_CHECKER_MODULES)"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
