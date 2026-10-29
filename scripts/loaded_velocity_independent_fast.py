"""081 scalar intent/route replay with independent preview and complete MPC audit."""

import math

import numpy as np
from scipy.spatial.transform import Rotation
from velocity_plan_compiled_audit import audit_plan
from velocity_plan_independent_audit import scalar_step
from verify_gripper_position_hold import GripperReplay


def compare(saved, expected):
    if isinstance(expected, dict):
        assert set(saved) == set(expected)
        for key, value in expected.items():
            compare(saved[key], value)
    elif expected is None or isinstance(expected, str):
        assert saved == expected
    elif isinstance(expected, bool):
        assert type(saved) is bool and saved == expected
    elif isinstance(expected, int):
        assert type(saved) is int and saved == expected
    else:
        np.testing.assert_allclose(saved, expected, rtol=0, atol=1e-10)


def preview(poses, reference, index, measured):
    stop = min(index + 6, len(reference) - 1)
    advanced = (
        all(reference[i + 1] >= reference[i] - 1e-12 for i in range(index, stop)) and reference[stop] > reference[index]
    )
    selected = stop if advanced else index
    proposal = dict(
        reference=float(reference[selected]),
        original_reference=float(reference[index]),
        selected_index=selected,
        current_index=index,
        available_end_index=len(reference) - 1,
        advanced=bool(advanced),
        preview_steps=6,
    )
    anchor = poses[index]
    segment = poses[index : selected + 1]
    position_span = max(float(np.linalg.norm(p[:3, 3] - anchor[:3, 3])) for p in segment)
    rotation_span = max(float(Rotation.from_matrix(anchor[:3, :3].T @ p[:3, :3]).magnitude()) for p in segment)
    position_error = float(np.linalg.norm(measured[:3, 3] - anchor[:3, 3]))
    rotation_error = float(Rotation.from_matrix(anchor[:3, :3].T @ measured[:3, :3]).magnitude())
    guards = dict(
        position_span=position_span <= 0.01,
        rotation_span=rotation_span <= 0.1,
        measured_position=position_error <= 0.01,
        measured_rotation=rotation_error <= 0.05,
    )
    accepted = advanced and all(guards.values())
    return dict(
        reference=float(reference[selected if accepted else index]),
        original_reference=float(reference[index]),
        advanced=bool(accepted),
        selected_index=selected if accepted else index,
        preview_proposal=proposal,
        guards=guards,
        position_span_m=position_span,
        rotation_span_rad=rotation_span,
        measured_position_error_m=position_error,
        measured_rotation_error_rad=rotation_error,
    )


class LoadedVelocityReplay:
    def __init__(self, parameters, initial_opening, config, envelope):
        self.parameters = parameters
        self.config = dict(config)
        self.envelope = envelope
        self.hold = GripperReplay(initial_opening, self.config, True)

    def step(self, saved, opening, motor, velocity, poses, reference, index, measured, previous, dt):
        assert dt == 0.05 and np.isfinite([opening, velocity]).all()
        motor = np.asarray(motor, dtype=float)
        assert (
            motor.shape == (2,)
            and np.isfinite(motor).all()
            and np.max(np.abs(motor)) <= 1
            and abs(float(motor.sum())) < 1e-10
        )
        expected_preview = preview(np.asarray(poses), np.asarray(reference), index, np.asarray(measured))
        compare(saved["preview"], expected_preview)
        target = expected_preview["reference"]
        width = float(np.clip((0.039 + 0.02 * (motor[0] - motor[1])) / 0.079, 0, 1))
        was_latched = self.hold.latched
        old_width = self.hold.width
        if previous < 0:
            old_width = None
        scalar_command = self.hold.step(target, opening, width, previous, dt)
        processed = was_latched or self.hold.latched
        if processed:
            possible_width = min(old_width, opening) if old_width is not None else opening
            if width > self.config["motor_limit_tolerance"]:
                possible_width = old_width
            release_level = min(
                max(opening, possible_width or 0.0) + self.config["release_margin"],
                1 - self.config["reference_hysteresis"] / 2,
            )
            excursion = target - self.hold.minimum
            if not self.hold.latched:
                state, reason = "position_tracking_release", "reference_rise_and_measured_clearance_confirmed"
            elif width > self.config["motor_limit_tolerance"]:
                state, reason = "closing", "continue_native_close_to_motor_limit"
            else:
                state, reason = "closed_motor_hold", "retain_closed_motor_target_not_observed_aperture"
        else:
            release_level = None
            excursion = 0.0
            state = "position_tracking" if scalar_command else "position_hold"
            reason = (
                "increase_known_motor_position"
                if scalar_command < 0
                else ("decrease_known_motor_position" if scalar_command > 0 else "retain_quantized_motor_position")
            )
        expected_latch = dict(
            command=scalar_command,
            state=state,
            reason=reason,
            reference=target,
            measured_opening=opening,
            motor_opening=width,
            normal_intent=self.hold.direction,
            closing_episode=self.hold.latched,
            holding_width=self.hold.width,
            reference_minimum=self.hold.minimum,
            reference_excursion=excursion,
            release_level=release_level,
            release_evidence_s=self.hold.confirm,
            evidence_interval_valid=True,
            reference_motor_error=target - width,
        )
        compare(saved["latch"], expected_latch)
        if self.hold.latched:
            assert saved["route"] == "native_close_load" and saved["plan"] is None
            assert saved["command"] == scalar_command
        else:
            assert saved["route"] == "model_tracking" and saved["plan"] is not None
            audit_plan(self.parameters, self.envelope, opening, motor, velocity, list(reference[index:]), saved["plan"])
            assert saved["command"] == saved["plan"]["command"]
        nq, nv, nm = scalar_step(self.parameters, opening, velocity, float(motor[0]), saved["command"])
        z = [
            (a - b) / s
            for a, b, s in zip(
                [opening, float(motor[0]), nm, velocity], self.parameters["mean"], self.parameters["scale"], strict=True
            )
        ]
        x = z + [z[i] * z[j] for i in range(4) for j in range(i, 4)] + [1.0]
        residual = [
            math.fsum(x[i] * self.parameters["coefficients"][i][column] for i in range(15)) for column in (0, 1)
        ]
        compare(
            saved["executed_response"],
            dict(
                next_opening=nq,
                next_period_velocity=nv,
                next_motor_target=nm,
                residual_opening=residual[0],
                residual_velocity=residual[1],
            ),
        )
        supported = abs(velocity) <= self.envelope and abs(nv) <= self.envelope
        compare(saved["executed_prediction_in_velocity_envelope"], bool(supported))
        assert set(saved) == {
            "command",
            "route",
            "latch",
            "preview",
            "plan",
            "executed_response",
            "executed_prediction_in_velocity_envelope",
        }
        return dict(route=saved["route"], command=saved["command"], executed_prediction_supported=bool(supported))
