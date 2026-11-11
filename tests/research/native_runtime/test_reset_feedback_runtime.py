"""A reset continuation has no fake prefix and a fixed eight-period budget."""

from pathlib import Path

import numpy as np
import pytest

from s2a.research.native_runtime.reset_feedback_runtime import ResetFeedbackRuntime
from tests.research.native_runtime.test_reset_endpoint import sample


def test_reset_continuation_binds_no_prior_command_and_exact_budget():
    expected, expected_receipt, chunk, controller = sample()
    runtime = ResetFeedbackRuntime(
        expected=expected, expected_receipt=expected_receipt, chunk=chunk, controller=controller, tool=np.eye(4)
    )

    def callback(*_args, **_kwargs):
        return None

    def fake_run(*args, **kwargs):
        assert len(args[3]) == 0 and args[3][-1] is None
        assert kwargs["registered_seed"] == 42 and kwargs["continuation"] is callback
        assert kwargs["continuation_budget"] == kwargs["continuation_limit"] == 8
        return {"behavioural_replay_verified": True, "previous_saved_command": None}

    runtime._run = fake_run
    result = runtime.run_feedback({}, {}, "", Path("."), {}, registered_seed=42, continuation=callback)
    assert result["behavioural_replay_verified"] is False and result["reset_observation_verified"] is True
    with pytest.raises(ValueError, match="continuation"):
        runtime.run_feedback({}, {}, "", Path("."), {}, registered_seed=42, continuation=None)
