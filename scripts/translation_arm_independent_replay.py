"""082 independent closed-form PD response and fixed-weight command replay."""

import math

import numpy as np
from scipy.spatial.transform import Rotation


class TranslationArmReplay:
    def __init__(self, interface, kp, kd, dt, *, compensate=True, inverse_weight=0.5, max_command_rate=None):
        assert compensate is True and inverse_weight == 0.5 and dt > 0
        self.interface = interface
        self.dt = dt
        self.rate = max_command_rate
        self.previous = np.minimum(np.maximum(np.zeros(6), interface.input_min), interface.input_max)
        response = []
        momentum = []
        for p, d in zip(kp, kd, strict=True):
            assert p > 0 and d > 0
            a = d / 2
            if math.isclose(a * a, p, rel_tol=1e-12, abs_tol=1e-12):
                f = math.exp(-a * dt)
                e00 = f * (1 + a * dt)
                e01 = f * dt
            elif p > a * a:
                b = math.sqrt(p - a * a)
                f = math.exp(-a * dt)
                e00 = f * (math.cos(b * dt) + a * math.sin(b * dt) / b)
                e01 = f * math.sin(b * dt) / b
            else:
                b = math.sqrt(a * a - p)
                r1, r2 = -a + b, -a - b
                e1, e2 = math.exp(r1 * dt), math.exp(r2 * dt)
                e00 = (-r2 * e1 + r1 * e2) / (r1 - r2)
                e01 = (e1 - e2) / (r1 - r2)
            response.append(1 - e00)
            momentum.append(e01)
        self.response = np.array(response)
        self.momentum = np.array(momentum)
        assert np.all(self.response > 1e-5)

    def convert(self, measured, desired, velocity):
        measured, desired, velocity = np.asarray(measured), np.asarray(desired), np.asarray(velocity)
        assert measured.shape == desired.shape == (4, 4) and velocity.shape == (6,)
        assert all(np.isfinite(v).all() for v in (measured, desired, velocity))
        error = np.r_[
            desired[:3, 3] - measured[:3, 3], Rotation.from_matrix(desired[:3, :3] @ measured[:3, :3].T).as_rotvec()
        ]
        inverse = (error - self.momentum * velocity) / self.response
        rotation_inverse_enabled = np.linalg.norm(error[3:]) <= 0.25
        if not rotation_inverse_enabled:
            inverse[3:] = error[3:]
        requested = np.r_[inverse[:3], 0.5 * error[3:] + 0.5 * inverse[3:]]
        i = self.interface
        raw = i.input_min + (requested - i.output_min) * (i.input_max - i.input_min) / (i.output_max - i.output_min)
        command = np.minimum(np.maximum(raw, i.input_min), i.input_max)
        saturated = bool(np.any(np.abs(raw - command) > 1e-12))
        limited = False
        if self.rate is not None:
            constrained = np.minimum(
                np.maximum(command, self.previous - self.rate * self.dt), self.previous + self.rate * self.dt
            )
            limited = bool(np.any(np.abs(constrained - command) > 1e-12))
            command = constrained
        self.previous = command.copy()
        return dict(
            command=command,
            saturated=saturated,
            rate_limited=limited,
            requested_delta=requested,
            rotation_inverse_enabled=bool(rotation_inverse_enabled),
        )
