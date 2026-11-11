"""Cumulative accounting includes failed physical work."""

import pytest

from scripts.simulation.runtime.run_early_tabletop_live_feedback_repair import cumulative_cost


def test_cumulative_cost_includes_capture_and_failed_attempt():
    config = {
        "prior_capture_native_steps": 12000,
        "prior_failed_native_steps": 6000,
        "repair_max_native_steps": 12800,
        "total_max_native_steps": 30800,
        "total_max_new_commands": 20,
    }
    result = cumulative_cost(12800, 16, config)
    assert result["experiment_native_steps"] == 30800 and result["experiment_new_commands"] == 20
    with pytest.raises(ValueError):
        cumulative_cost(12801, 16, config)
