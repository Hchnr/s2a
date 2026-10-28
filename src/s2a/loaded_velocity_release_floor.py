"""081 explicit native-loading versus model-tracking routes using causal state."""

from dataclasses import asdict, dataclass

import numpy as np

from s2a.coupled_gripper_preview import coupled_opening_preview
from s2a.gripper_release_floor import GripperPositionHold
from s2a.gripper_tracking import panda_opening_from_motor_target
from s2a.velocity_model_planner import VelocityModelPlanner


@dataclass(frozen=True)
class LoadedVelocityResult:
    command: float
    route: str
    latch: dict
    preview: dict
    plan: dict | None
    executed_response: dict
    executed_prediction_in_velocity_envelope: bool


class LoadedVelocityControl:
    def __init__(self, parameters, initial_opening, *, velocity_envelope):
        self.planner = VelocityModelPlanner(parameters, velocity_envelope=velocity_envelope)
        self.latch = GripperPositionHold(initial_opening)

    def update(
        self,
        opening,
        motor_cache,
        period_velocity,
        generated_poses,
        generated_openings,
        current_index,
        measured_tcp,
        *,
        previous_command,
        dt,
    ):
        motor = np.asarray(motor_cache, dtype=float)
        if (
            motor.shape != (2,)
            or not np.isfinite(motor).all()
            or np.any(np.abs(motor) > 1)
            or abs(float(motor.sum())) > 1e-10
        ):
            raise ValueError("Actual antisymmetric Panda motor cache required")
        if not np.isfinite(period_velocity) or dt != 0.05:
            raise ValueError("Current finite velocity and fixed20Hz interval required")
        preview = coupled_opening_preview(
            generated_poses,
            generated_openings,
            current_index,
            measured_tcp,
            preview_steps=6,
            position_span_max_m=0.01,
            rotation_span_max_rad=0.1,
            measured_position_max_m=0.01,
            measured_rotation_max_rad=0.05,
        )
        latch = self.latch.update(
            preview["reference"],
            opening,
            motor_opening=panda_opening_from_motor_target(motor),
            previous_command=previous_command,
            dt=dt,
        )
        if latch.closing_episode:
            command = latch.command
            route = "native_close_load"
            plan = None
        else:
            selected = self.planner.plan(
                opening, motor, period_velocity, np.asarray(generated_openings)[current_index:]
            )
            command = selected.command
            route = "model_tracking"
            plan = asdict(selected)
        executed = self.planner.model.predict(opening, float(motor[0]), period_velocity, command)
        supported = (
            abs(period_velocity) <= self.planner.velocity_envelope
            and abs(executed.next_period_velocity) <= self.planner.velocity_envelope
        )
        return LoadedVelocityResult(command, route, asdict(latch), preview, plan, asdict(executed), bool(supported))
