import numpy as np
import pytest

from s2a.terminal_readout_policy import TerminalReadoutPolicy, TerminalStepMixin
from s2a.wam_clean_readout import CleanResidualReadout


def heads():
    base = CleanResidualReadout(np.ones((1291, 10)), np.zeros(1290), 2.0, 0.1 / 1290)
    changed = base.coefficients.copy()
    changed[:, [0, 1, 2, 9]] += 0.1
    terminal = CleanResidualReadout(changed, base.center, base.scale, base.ridge)
    return base, terminal


def test_only_real_final_index_selects_terminal():
    base, terminal = heads()
    policy = TerminalReadoutPolicy(base, terminal)
    assert all(policy.at_step(i) is base for i in range(34))
    assert policy.at_step(34) is terminal
    for bad in (-1, 35, True, 34.0, None):
        with pytest.raises(ValueError):
            policy.at_step(bad)


def test_rotation_change_rejected():
    base, terminal = heads()
    changed = terminal.coefficients.copy()
    changed[0, 3] += 1
    with pytest.raises(ValueError, match="rotation"):
        TerminalReadoutPolicy(base, CleanResidualReadout(changed, base.center, base.scale, base.ridge))


def test_bridge_uses_actual_record_and_restores_readout_on_exception():
    class Parent:
        def _finish_velocity(self, raw):
            assert self.readout is terminal
            raise RuntimeError("original bridge failure")

    class Bridge(TerminalStepMixin, Parent):
        pass

    base, terminal = heads()
    bridge = Bridge()
    bridge.readout = base
    bridge.terminal_policy = TerminalReadoutPolicy(base, terminal)
    bridge._step = {"record": {"step_index": 34}}
    with pytest.raises(RuntimeError, match="original bridge"):
        bridge._finish_velocity(object())
    assert bridge.readout is base
