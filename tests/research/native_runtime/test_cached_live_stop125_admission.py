"""The static stop diagnostic may use only completed current information."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import cached_live_stop125_admission as admission
from scripts.verification.reference_ports import audit_causal_stream_live_cached as audit


def test_actual_stopped_state_and_past_only_inputs_are_bound():
    inputs = admission.load_inputs()
    stopped = inputs["stop_inputs"]
    assert set(stopped) == {"state", "previous", "shifted_seed", "past_inner_error_rad"}
    assert stopped["state"].shape == (79,) and stopped["shifted_seed"].shape == (8, 6)
    assert stopped["past_inner_error_rad"].shape == (25, 7)
    assert inputs["targets"].shape == (40, 4, 4)
    assert np.all(np.diff(stopped["past_inner_error_rad"], axis=0) >= 0)


def test_a_changed_stopped_state_is_not_admitted():
    summaries = [
        prefix.read({"path": str(audit.RUN / phase / "result.json"), "sha256": sha})
        for phase, sha in (("producer", audit.PRODUCER_SHA), ("checker", audit.CHECKER_SHA))
    ]
    producer, checker = [prefix.read(summary["branch_ref"]) for summary in summaries]
    producer = copy.deepcopy(producer)
    producer["runtime"]["continuation"]["segments"][-1]["continuation"]["periods"][-1]["actual_state"][1] += 0.1
    with pytest.raises(AssertionError):
        admission.stopped_inputs(producer, checker)


@pytest.mark.parametrize(("key", "value"), [("physics_steps", 1), ("max_optimizer_calls", 5), ("stop_step", 126)])
def test_unregistered_work_is_rejected(key, value):
    config = copy.deepcopy(admission.FIXED)
    config[key] = value
    with pytest.raises(ValueError, match="diagnostic budget"):
        admission.validate_config(config)
