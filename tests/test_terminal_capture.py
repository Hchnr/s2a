import numpy as np

from s2a.terminal_capture import TerminalPolicyBridgeMixin
from s2a.terminal_readout_policy import TerminalReadoutPolicy
from s2a.wam_clean_readout import CleanResidualReadout


def test_candidate_then_raw_then_zero_do_not_leak_across_requests():
    class Parent:
        def _finish_velocity(self, raw):
            return self.readout

    class Bridge(TerminalPolicyBridgeMixin, Parent):
        pass

    base = CleanResidualReadout(np.ones((1291, 10)), np.zeros(1290), 1.0, 0.1 / 1290)
    coefficients = base.coefficients.copy()
    coefficients[:, 9] += 0.1
    terminal = CleanResidualReadout(coefficients, base.center, base.scale, base.ridge)
    policy = TerminalReadoutPolicy(base, terminal)
    bridge = Bridge()
    bridge.readout = policy
    for step in range(35):
        bridge._step = {"record": {"step_index": step}}
        assert bridge._finish_velocity(object()) is (terminal if step == 34 else base)
        assert bridge.readout is policy
    bridge.readout = None
    assert bridge._finish_velocity(object()) is None
    assert bridge.terminal_policy is None
    zero = CleanResidualReadout(np.zeros((1291, 10)), np.zeros(1290), 1.0, 0.1)
    bridge.readout = zero
    assert bridge._finish_velocity(object()) is zero
