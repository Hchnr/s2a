"""A longer reset run preserves no-prefix semantics and the fixed budget."""

from pathlib import Path

import numpy as np
import pytest

from s2a.research.native_runtime.reset_multi_window_runtime import ResetMultiWindowRuntime
from tests.research.native_runtime.test_reset_endpoint import sample


def test_longer_reset_uses_exact_48_period_budget_without_prior_commands():
    expected, endpoint, chunk, controller = sample()
    runtime = ResetMultiWindowRuntime(
        expected=expected, expected_receipt=endpoint, chunk=chunk, controller=controller, tool=np.eye(4)
    )

    def continuation(*_args, **_kwargs):
        return None

    def fake_run(*args, **kwargs):
        assert len(args[3]) == 0 and args[3][-1] is None
        assert kwargs["continuation_budget"] == kwargs["continuation_limit"] == 48
        assert kwargs["registered_seed"] == 42 and kwargs["continuation"] is continuation
        return {"behavioural_replay_verified": True}

    runtime._run = fake_run
    result = runtime.run_feedback({}, {}, "", Path("."), {}, registered_seed=42, continuation=continuation)
    assert result["reset_observation_verified"] is True and result["behavioural_replay_verified"] is False
    with pytest.raises(ValueError, match="continuation"):
        runtime.run_feedback({}, {}, "", Path("."), {}, registered_seed=42, continuation=None)
