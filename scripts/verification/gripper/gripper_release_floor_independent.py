"""Independent scalar state machine for the explicit release-floor candidate."""


class GripperReplay:
    """Scalar protocol replay, independent of both adapter implementations."""

    def __init__(self, initial, config, position):
        self.c, self.position = config, position
        self.extreme, self.direction = initial, -1.0
        self.latched, self.minimum, self.width = False, initial, None
        self.confirm, self.last_evidence = 0.0, False

    def step(self, reference, measured, motor, previous, dt):
        c = self.c
        if self.direction < 0:
            self.extreme = max(self.extreme, reference)
            if reference < self.extreme - c["reference_hysteresis"]:
                self.direction, self.extreme = 1.0, reference
        else:
            self.extreme = min(self.extreme, reference)
            if reference > self.extreme + c["reference_hysteresis"]:
                self.direction, self.extreme = -1.0, reference
        valid = dt <= c["max_evidence_interval_s"] + 1e-12
        if not valid or previous < 0:
            self.confirm, self.last_evidence, self.width = 0.0, False, None
        if not self.latched and self.direction > 0 and (not self.position or reference <= c["close_threshold"]):
            self.latched, self.minimum = True, reference
        if self.latched:
            self.minimum = min(self.minimum, reference)
            if motor <= c["motor_limit_tolerance"]:
                self.width = measured if self.width is None else min(self.width, measured)
            limit = min(max(measured, self.width or 0.0) + c["release_margin"], 1 - c["reference_hysteresis"] / 2)
            limit = max(c["release_reference_minimum"], limit)
            release = (
                valid
                and self.direction < 0
                and reference >= limit
                and reference - self.minimum >= min(c["release_excursion"], (1 - self.minimum) / 2)
            )
            self.confirm = self.confirm + dt if release and self.last_evidence else 0.0
            self.last_evidence = release
            if self.confirm + 1e-12 >= c["release_confirmation_s"]:
                self.latched, self.width = False, None
                return self.position_command(reference, motor) if self.position else -1.0
            return 1.0 if motor > c["motor_limit_tolerance"] else 0.0
        self.confirm, self.last_evidence = 0.0, False
        if self.position:
            return self.position_command(reference, motor)
        return -1.0 if motor < 1 - c["motor_limit_tolerance"] else 0.0

    def position_command(self, reference, motor):
        if reference > motor + self.c["motor_deadband"]:
            return -1.0
        if reference < motor - self.c["motor_deadband"]:
            return 1.0
        return 0.0
