"""Lifecycle checks prevent incomplete workers from being counted as evidence."""

import copy

import pytest

from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.verification.response_models.audit_segmented_reset_equivalence import validate_workers


def test_all_workers_must_finish_in_order_and_be_waited():
    workers = [
        dict(key=original.branch_key(i, "feedback"), returncode=0, actual_wait=True, reaped=True, signal_sent=False)
        for i in range(12)
    ]
    lifecycle = dict(launch_error=None, workers=workers)
    validate_workers(lifecycle)
    for key, value in (("actual_wait", False), ("reaped", False), ("returncode", 1), ("signal_sent", True)):
        bad = copy.deepcopy(lifecycle)
        bad["workers"][5][key] = value
        with pytest.raises(ValueError, match="twelve registered workers"):
            validate_workers(bad)
    bad = copy.deepcopy(lifecycle)
    bad["workers"].reverse()
    with pytest.raises(ValueError, match="twelve registered workers"):
        validate_workers(bad)
