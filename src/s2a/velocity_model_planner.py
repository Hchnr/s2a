"""080 exhaustive finite native-command planning with a frozen079 response model."""

from dataclasses import dataclass

import numpy as np

from s2a.velocity_residual_response import VelocityResidualResponse, design_features

COMMAND_ORDER = (0.0, -1.0, 1.0)


@dataclass(frozen=True)
class VelocityPlan:
    command: float
    sequence: tuple
    predicted_openings: tuple
    predicted_period_velocities: tuple
    predicted_motor_targets: tuple
    reference_endpoints: tuple
    horizon: int
    reference_available_count: int
    cost: float
    valid_sequences: int
    rejected_nonfinite_sequences: int
    rejected_velocity_sequences: int


def fit_velocity_envelope(raw, residual_targets):
    raw = np.asarray(raw, dtype=float)
    target = np.asarray(residual_targets, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 4 or not len(raw) or target.shape != (len(raw), 2):
        raise ValueError("079 training rows and two residual targets required")
    if not np.isfinite(raw).all() or not np.isfinite(target).all():
        raise ValueError("Finite fit-only velocity support required")
    bound = 1.5 * float(np.max(np.abs(np.r_[raw[:, 3], raw[:, 3] + target[:, 1]])))
    if not np.isfinite(bound):
        raise ValueError("Finite velocity envelope required")
    return bound


class VelocityModelPlanner:
    def __init__(self, parameters, *, velocity_envelope):
        self.model = VelocityResidualResponse(parameters)
        if not np.isfinite(velocity_envelope) or velocity_envelope < 0:
            raise ValueError("Finite nonnegative fit velocity envelope required")
        self.velocity_envelope = float(velocity_envelope)

    def plan(self, opening, motor_cache, period_velocity, generated_openings):
        reference = np.asarray(generated_openings, dtype=float)
        motor = np.asarray(motor_cache, dtype=float)
        if (
            reference.ndim != 1
            or not len(reference)
            or not np.isfinite(reference).all()
            or np.any((reference < 0) | (reference > 1))
        ):
            raise ValueError("Already generated finite opening endpoints required")
        if motor.shape not in ((1,), (2,)) or not np.isfinite(motor).all() or np.any(np.abs(motor) > 1):
            raise ValueError("Current normalized motor cache required")
        if len(motor) == 2 and abs(float(motor.sum())) > 1e-10:
            raise ValueError("Panda motor targets must be antisymmetric")
        if not np.isfinite([opening, period_velocity]).all() or not 0 <= opening <= 1:
            raise ValueError("Current measured opening and velocity required")
        if abs(period_velocity) > self.velocity_envelope:
            raise ValueError("Current velocity outside fit support envelope")
        horizon = min(8, len(reference))
        q, v, m, cost = (np.array([value], dtype=float) for value in (opening, period_velocity, motor[0], 0.0))
        sequence, qtrace, vtrace, mtrace = (np.empty((1, 0)) for _ in range(4))
        nonfinite_rejected = velocity_rejected = 0
        for depth in range(horizon):
            q, v, m, cost = (np.repeat(value, 3) for value in (q, v, m, cost))
            sequence, qtrace, vtrace, mtrace = (
                np.repeat(value, 3, axis=0) for value in (sequence, qtrace, vtrace, mtrace)
            )
            command = np.tile(COMMAND_ORDER, len(q) // 3)
            nextm = m.copy()
            for _ in range(25):
                nextm = np.clip(nextm - 0.01 * command, -1, 1)
            with np.errstate(over="ignore", invalid="ignore"):
                residual = (
                    design_features(np.column_stack([q, m, nextm, v]), self.model.mean, self.model.scale)
                    @ self.model.coefficients
                )
                rawq, nextv = q + v + residual[:, 0], v + residual[:, 1]
            finite = np.isfinite(rawq) & np.isfinite(nextv)
            supported = finite & (np.abs(nextv) <= self.velocity_envelope)
            descendants = 3 ** (horizon - depth - 1)
            nonfinite_rejected += int(np.sum(~finite)) * descendants
            velocity_rejected += int(np.sum(finite & ~supported)) * descendants
            if not supported.any():
                raise ValueError("No finite supported native-command sequence")
            nextq = np.clip(rawq[supported], 0, 1)
            nextv, nextm, command = nextv[supported], nextm[supported], command[supported]
            cost = cost[supported] + (nextq - reference[depth]) ** 2 + 0.0001 * command**2
            sequence = np.column_stack([sequence[supported], command])
            qtrace = np.column_stack([qtrace[supported], nextq])
            vtrace = np.column_stack([vtrace[supported], nextv])
            mtrace = np.column_stack([mtrace[supported], nextm])
            q, v, m = nextq, nextv, nextm
        chosen = int(np.argmin(cost))
        assert len(cost) + nonfinite_rejected + velocity_rejected == 3**horizon
        return VelocityPlan(
            float(sequence[chosen, 0]),
            tuple(sequence[chosen]),
            tuple(qtrace[chosen]),
            tuple(vtrace[chosen]),
            tuple(mtrace[chosen]),
            tuple(reference[:horizon]),
            horizon,
            len(reference),
            float(cost[chosen]),
            len(cost),
            nonfinite_rejected,
            velocity_rejected,
        )
