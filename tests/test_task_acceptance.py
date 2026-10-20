import numpy as np
import pytest

from s2a.task_acceptance import chunk_opening_diagnostics, scene_bootstrap, task_outcome


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


def test_closure_deferral_depends_on_a_real_unexecuted_suffix():
    future_close = np.ones((40, 128))
    future_close[20:, 9] = 0
    already_closed = np.ones((40, 128))
    already_closed[0, 9] = 0
    always_open = np.ones((40, 128))
    chunks = [future_close, already_closed, always_open]
    short = chunk_opening_diagnostics(chunks, execution_horizon=8)
    assert short["deferred_closure_chunks"] == 1
    assert short["executed_prefix_min"] == short["full_prediction_min"] == 0
    # Neither a closed prefix nor an all-open chunk has a deferred closure
    # when all forty endpoints execute. This used to reduce an empty suffix.
    full = chunk_opening_diagnostics(chunks, execution_horizon=40)
    assert full["deferred_closure_chunks"] == 0
    assert full["chunks"] == 3
    assert full["executed_prefix_min"] == full["full_prediction_min"] == 0
    assert chunk_opening_diagnostics([future_close], execution_horizon=16)["deferred_closure_chunks"] == 1
    assert chunk_opening_diagnostics([future_close], execution_horizon=24)["deferred_closure_chunks"] == 0
