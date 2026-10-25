import numpy as np

from s2a.terminal_history_comparison import INPUTS, OUTPUTS, compare_point


def arrays():
    return {name: np.zeros((40, 1280 if name == "hidden" else 10)) for name in INPUTS + OUTPUTS}


def test_terminal_changes_only_allow_four_output_columns_not_inputs():
    old = arrays()
    new = {k: v.copy() for k, v in old.items()}
    for key in OUTPUTS:
        new[key][:, [0, 1, 2, 9]] = 1
    assert compare_point(new, old, 34)["passed"]
    assert not compare_point(new, old, 33)["passed"]
    new["hidden"][0, 0] = 1
    assert not compare_point(new, old, 34)["passed"]


def test_rotation_change_is_never_allowed_at_terminal():
    old = arrays()
    new = {k: v.copy() for k, v in old.items()}
    new["deployed_velocity_active10"][0, 3] = 1
    assert not compare_point(new, old, 34)["passed"]
