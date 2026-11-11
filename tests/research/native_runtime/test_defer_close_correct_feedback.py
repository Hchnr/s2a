"""The fixed physical diagnostic changes exactly one registered command."""

import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import run_defer_close_correct as feedback


def test_only_offset_five_changes_and_original_inputs_remain_untouched():
    commands = np.zeros((8, 7))
    commands[5, 6] = 1.0
    original = commands.copy()
    for offset in range(8):
        proposal = {"deferred": True, "command": 0.0} if offset == 5 else None
        candidate, changed = feedback.diagnostic_command(commands, offset, "deferred", proposal)
        expected = commands[offset].copy()
        if offset == 5:
            expected[6] = 0
        np.testing.assert_array_equal(candidate, expected)
        assert changed is (offset == 5)
        np.testing.assert_array_equal(feedback.diagnostic_command(commands, offset, "baseline")[0], commands[offset])
    np.testing.assert_array_equal(commands, original)
    with pytest.raises(ValueError, match="fixed"):
        feedback.diagnostic_command(commands, 5, "deferred", {"deferred": False, "command": 0.0})


def test_checker_entry_does_not_import_candidate_module():
    source = (
        "import sys; import scripts.simulation.runtime.run_defer_close_correct as m; "
        "m.checker_isolation(); assert m.CANDIDATE_MODULE not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", source], env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
