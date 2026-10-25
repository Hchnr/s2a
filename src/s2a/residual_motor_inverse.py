"""Frozen073 residual motor predictor; explicit causal state and generated reference."""

from dataclasses import dataclass

import numpy as np

from s2a.state_conditioned_gripper_inverse import design_features, raw_features, reachable_command


@dataclass(frozen=True)
class ResidualMotorResult:
    command: float
    predicted_motor_target: float
    selected_motor_target: float
    previous_opening_delta: float
    available: tuple


class ResidualMotorInverse:
    """No hidden episode state: caller supplies only the preceding measured opening."""

    def __init__(self, parameters):
        if (
            parameters.get("target") != "next_motor_minus_current_motor"
            or parameters.get("baseline") != "current_motor"
        ):
            raise ValueError("Frozen residual target and current motor baseline required")
        self.mean = np.array(parameters["mean"], dtype=float, copy=True)
        self.scale = np.array(parameters["scale"], dtype=float, copy=True)
        self.coefficients = np.array(parameters["coefficients"], dtype=float, copy=True)
        if self.coefficients.shape != (40,) or not np.isfinite(self.coefficients).all():
            raise ValueError("Forty finite coefficients required")
        # Validate all normalization fields before the first physical command.
        design_features(np.zeros(7), np.zeros(4, dtype=bool), self.mean, self.scale)
        for value in (self.mean, self.scale, self.coefficients):
            value.setflags(write=False)

    def update(self, opening, motor_cache, generated_openings, *, previous_opening):
        cache = np.asarray(motor_cache, dtype=float)
        if cache.shape not in ((1,), (2,)) or not np.isfinite(cache).all() or np.any(np.abs(cache) > 1):
            raise ValueError("Actual normalized one/two-component motor cache required")
        if cache.shape == (2,) and abs(float(cache.sum())) > 1e-10:
            raise ValueError("Panda motor targets must be antisymmetric")
        if previous_opening is None:
            delta = 0.0
        else:
            if not np.isfinite(previous_opening) or not 0 <= previous_opening <= 1:
                raise ValueError("Previous measured opening must lie in [0,1]")
            delta = float(opening - previous_opening)
        raw, mask = raw_features(opening, float(cache[0]), delta, generated_openings)
        features = design_features(raw, mask, self.mean, self.scale)
        prediction = float(np.clip(cache[0] + features @ self.coefficients, -1, 1))
        action = reachable_command(float(cache[0]), prediction)
        return ResidualMotorResult(
            action["command"], prediction, action["selected_motor_target"], delta, tuple(bool(v) for v in mask)
        )
