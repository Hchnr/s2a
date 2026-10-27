"""Attach terminal policy to the existing capture bridge without duplicating it."""

from s2a.terminal_readout_policy import TerminalReadoutPolicy, TerminalStepMixin


class TerminalPolicyBridgeMixin(TerminalStepMixin):
    """Compose before the original ConversionWitness in the sampler MRO.

    Every request's existing readout assignment owns its policy. Explicitly reset
    terminal_policy for raw/zero so a previous candidate cannot leak into them.
    """

    def _finish_velocity(self, raw):
        self.terminal_policy = self.readout if isinstance(self.readout, TerminalReadoutPolicy) else None
        return super()._finish_velocity(raw)
