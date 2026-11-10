"""The dispatch gate reacts to completed measured samples, not future plans."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.completed_buffer_gate import CompletedBufferGate


def inputs():
    return np.zeros(7), np.tile([-1.0, 1.0], (7, 1)), np.zeros((6, 7)), np.zeros(7)


def test_inner_loss_blocks_the_next_dispatch_after_completion():
    gate = CompletedBufferGate(initial_inner_loss=False)
    q, limits, jac, previous = inputs()
    calls = []

    def choose(*args):
        calls.append(args[-1])
        return {"command": np.zeros(7), "newly_selected": True}

    assert "command" in gate.select(q, limits, jac, previous, 0, choose)
    with pytest.raises(ValueError, match="pending"):
        gate.select(q, limits, jac, previous, 1, choose)
    samples = np.zeros((25, 7))
    samples[17, 3] = 0.96
    assert gate.complete(0, samples)["below_buffer"]
    assert gate.select(q, limits, jac, previous, 1, choose)["stop"]["reason"] == "observed_joint_buffer_loss"
    assert calls == [0]


def test_current_and_initial_losses_stop_before_calling_the_candidate():
    q, limits, jac, previous = inputs()

    def choose(*_):
        pytest.fail("unsafe state must not call candidate")

    assert "stop" in CompletedBufferGate(initial_inner_loss=True).select(q, limits, jac, previous, 0, choose)
    q[0] = 0.951
    assert "stop" in CompletedBufferGate(initial_inner_loss=False).select(q, limits, jac, previous, 0, choose)


def test_stopped_candidate_never_stages_a_completed_period():
    gate = CompletedBufferGate(initial_inner_loss=False)
    q, limits, jac, previous = inputs()
    result = gate.select(q, limits, jac, previous, 0, lambda *_: {"stop": {"reason": "quality_no_witness"}})
    assert result["stop"]["reason"] == "quality_no_witness" and gate.pending is None
    with pytest.raises(ValueError, match="staged"):
        gate.complete(0, np.zeros((25, 7)))
