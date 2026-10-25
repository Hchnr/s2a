"""Versioned bridge fix: fingerprint both heads and the explicit dispatch rule.

A terminal policy is not a single CleanResidualReadout. Preserve the parent's
raw/zero identity path and its request-before/after mutation guard.
"""

from s2a.terminal_capture import TerminalPolicyBridgeMixin
from s2a.terminal_readout_policy import TerminalReadoutPolicy
from s2a.wam_clean_capture import json_sha


class TerminalIdentityBridgeMixin(TerminalPolicyBridgeMixin):
    def _readout_sha(self):
        if not isinstance(self.readout, TerminalReadoutPolicy):
            return super()._readout_sha()
        return json_sha(
            dict(
                schema_version="s2a.terminal-readout-policy.v1",
                baseline=self.readout.baseline.to_parameters(),
                terminal=self.readout.terminal.to_parameters(),
                terminal_step_index=34,
                total_steps=35,
            )
        )
