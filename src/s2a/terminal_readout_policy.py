"""Uniform causal dispatch for the single plan057/058 terminal candidate."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerminalReadoutPolicy:
    baseline: object
    terminal: object

    def __post_init__(self):
        if (
            self.baseline.coefficients.shape != (1291, 10)
            or self.terminal.coefficients.shape != (1291, 10)
            or not np.array_equal(self.baseline.coefficients[:, 3:9], self.terminal.coefficients[:, 3:9])
            or not np.array_equal(self.baseline.center, self.terminal.center)
            or self.baseline.scale != self.terminal.scale
            or self.baseline.ridge != self.terminal.ridge
        ):
            raise ValueError("Terminal policy must preserve baseline rotation and feature transform")

    def at_step(self, step_index):
        if type(step_index) is not int or not 0 <= step_index < 35:
            raise ValueError("Actual integer scheduler index in [0,34] required")
        return self.terminal if step_index == 34 else self.baseline


class TerminalStepMixin:
    """Select using the actual capture bridge record, never a separate counter.

    Compose before ConversionWitness in MRO. Sampler sets terminal_policy once
    per request; None leaves the existing zero/raw readout path intact.
    Parent bridge retains all original denoise/head/sigma/source checks.
    """

    terminal_policy = None

    def _finish_velocity(self, raw):
        if self.terminal_policy is None:
            return super()._finish_velocity(raw)
        original = self.readout
        self.readout = self.terminal_policy.at_step(self._step["record"]["step_index"])
        try:
            return super()._finish_velocity(raw)
        finally:
            self.readout = original
