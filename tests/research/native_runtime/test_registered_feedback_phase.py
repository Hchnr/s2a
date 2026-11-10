"""Feedback phases must not close before every owned worker is reaped."""

import pytest

from scripts.simulation.runtime.registered_feedback_phase import require_closed_workers


def test_worker_set_requires_matching_identities_and_real_reaping():
    workers = [dict(key=name, returncode=0, actual_wait=True, reaped=True, signal_sent=False) for name in ("a", "b")]
    require_closed_workers(dict(launch_error=None, workers=workers), ["a", "b"])
    with pytest.raises(ValueError, match="registered feedback worker"):
        require_closed_workers(dict(launch_error=None, workers=workers), ["b", "a"])
    workers[0]["reaped"] = False
    with pytest.raises(ValueError, match="reaped"):
        require_closed_workers(dict(launch_error=None, workers=workers), ["a", "b"])
