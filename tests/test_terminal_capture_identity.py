import numpy as np

from s2a.terminal_capture_identity import TerminalIdentityBridgeMixin
from s2a.terminal_readout_policy import TerminalReadoutPolicy
from s2a.wam_clean_capture import CleanCaptureBridge, json_sha
from s2a.wam_clean_readout import CleanResidualReadout


class ActualBridge(TerminalIdentityBridgeMixin, CleanCaptureBridge):
    pass


def bridge():
    # Invoke the actual parent's identity method without loading a model.
    result = object.__new__(ActualBridge)
    base = CleanResidualReadout(np.ones((1291, 10)), np.zeros(1290), 1.0, 0.1 / 1290)
    changed = base.coefficients.copy()
    changed[:, 9] += 0.1
    terminal = CleanResidualReadout(changed, base.center, base.scale, base.ridge)
    result.readout = TerminalReadoutPolicy(base, terminal)
    return result


def test_real_parent_request_identity_accepts_policy_and_detects_each_head_mutation():
    result = bridge()
    initial = result._readout_sha()
    assert isinstance(initial, str) and len(initial) == 64
    assert initial == result._readout_sha()
    previous = result.readout
    changed = previous.terminal.coefficients.copy()
    changed[0, 9] += 0.1
    result.readout = TerminalReadoutPolicy(
        previous.baseline,
        CleanResidualReadout(changed, previous.terminal.center, previous.terminal.scale, previous.terminal.ridge),
    )
    updated = result._readout_sha()
    assert initial != updated
    previous = result.readout
    changed = previous.baseline.coefficients.copy()
    changed[0, 0] += 0.1
    result.readout = TerminalReadoutPolicy(
        CleanResidualReadout(changed, previous.baseline.center, previous.baseline.scale, previous.baseline.ridge),
        previous.terminal,
    )
    assert updated != result._readout_sha()


def test_actual_parent_raw_and_zero_identity_remain_unchanged():
    result = bridge()
    result.readout = None
    assert result._readout_sha() is None
    zero = CleanResidualReadout(np.zeros((1291, 10)), np.zeros(1290), 1.0, 0.1)
    result.readout = zero
    assert result._readout_sha() == json_sha(zero.to_parameters())
