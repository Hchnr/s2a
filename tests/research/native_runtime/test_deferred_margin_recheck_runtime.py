"""The WAM recheck preserves its one-event and physical-budget registration."""

import json

import pytest

from scripts.simulation.runtime import bounded_reference_choice as runtime
from scripts.simulation.runtime import run_refresh_margin_physical as previous
from scripts.simulation.runtime.run_deferred_margin_recheck import (
    CONFIG,
    OUTPUT,
    comparison_with_gate,
    configured_phase,
    validate_config,
)


def test_registered_trigger_and_limits_cannot_be_changed():
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    for key, value in (
        ("deferred_trigger_step", 16),
        ("max_deferred_triggers", 2),
        ("max_control_steps_per_phase", 600),
        ("world_model_calls", 1),
        ("automatic_retry", 1),
    ):
        changed = dict(config, **{key: value})
        with pytest.raises(ValueError):
            validate_config(changed)


def test_diagnostic_bindings_do_not_mutate_the_recorded_runtime_or_owner():
    original_stop = runtime.run_comparison.__globals__["quality_stop"]
    original_output = previous.phase_run.__globals__["OUTPUT"]

    def gate(*_args):
        return None

    comparison = comparison_with_gate(gate)
    owner = configured_phase()
    assert comparison.__globals__["quality_stop"] is gate
    assert runtime.run_comparison.__globals__["quality_stop"] is original_stop
    assert owner.__globals__["OUTPUT"] == OUTPUT
    assert previous.phase_run.__globals__["OUTPUT"] == original_output != OUTPUT
