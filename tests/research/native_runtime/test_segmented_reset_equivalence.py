"""The bounded physical equivalence gate rejects changed actions and sources."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import run_segmented_reset_equivalence as study


def test_equivalence_budget_rejects_more_periods_and_implicit_retries():
    config = json.loads(study.CONFIG.read_text())
    study.validate_config(config)
    for key, value in (
        ("continuation_steps", 9),
        ("world_model_calls", False),
        ("automatic_retry", True),
        ("case_indices", [False, *range(1, 12)]),
        ("comparison_relative_tolerance", 0),
    ):
        with pytest.raises(ValueError):
            study.validate_config(dict(config, **{key: value}))


def test_changed_or_stopped_action_is_rejected_before_dispatch():
    expected = {"command": np.zeros(7)}
    candidate = {"command": np.zeros(7), "newly_selected": True}
    assert study.selected_command(candidate, expected) is candidate
    changed = {"command": np.array([0, 0, 0, 0, 0, 0, 1e-12]), "newly_selected": True}
    with pytest.raises(AssertionError):
        study.selected_command(changed, expected)
    with pytest.raises(ValueError, match="original completed action"):
        study.selected_command({"stop": {"reason": "no_witness"}}, expected)


def test_checker_entry_does_not_import_any_candidate_or_simulator():
    code = (
        "from scripts.simulation.runtime import run_segmented_reset_equivalence as s; "
        "s.checker_isolation(); s.original.require_no_simulator_import()"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
