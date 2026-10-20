import numpy as np
import pytest

from s2a.task_acceptance import scene_bootstrap, task_outcome


def test_transient_success_and_incomplete_budget_cannot_pass_stability():
    transient = [False, True, True, False, True]
    result = task_outcome(transient, expected_steps=5, stable_steps=2)
    assert result["ever"] and result["terminal"] and not result["stable"]
    assert result["trailing_success_steps"] == 1
    with pytest.raises(ValueError, match="budgeted"):
        task_outcome(transient[:-1], expected_steps=5, stable_steps=2)
    assert task_outcome([False, False, True, True, True], expected_steps=5, stable_steps=3)["stable"]


def test_repeated_identical_seeds_do_not_artificially_narrow_scene_interval():
    values = np.array([[[True], [False], [False], [True]], [[False], [True], [False], [False]]])
    single = scene_bootstrap(values)
    repeated = scene_bootstrap(np.repeat(values, 20, axis=2))
    assert single["interval95"] == repeated["interval95"]
    assert single["scene_count"] == repeated["scene_count"] == 8
    assert repeated["trial_count"] == 160
