"""Causal state transitions motivated by actual Panda/WAM failures."""

import inspect

import numpy as np
import pytest

from s2a.gripper_position_hold import GripperPositionHold, GripperPositionHoldConfig


def update(adapter, reference, *, measured=0.53, motor=0.0, previous=0.0, dt=0.05):
    return adapter.update(reference, measured, motor_opening=motor, previous_command=previous, dt=dt)


def test_high_reference_decline_does_not_command_full_stroke():
    adapter = GripperPositionHold(1.0)
    motor = 1.0
    commands = []
    for reference in [1.0, 0.99, 0.97, 0.95, 0.93, 0.922] + [0.922] * 15:
        result = update(adapter, reference, measured=motor, motor=motor)
        assert not result.closing_episode
        commands.append(result.command)
        motor = float(np.clip(motor - result.command * 0.1265822784810128, 0, 1))
    assert commands.count(1.0) == 1
    assert motor == pytest.approx(0.8734177215189872)
    assert commands[-10:] == [0.0] * 10


def test_quantization_does_not_reverse_for_fixed_intermediate_reference():
    adapter = GripperPositionHold(0.9)
    motor = 0.9
    for _ in range(20):
        result = update(adapter, 0.77, measured=motor, motor=motor)
        motor = float(np.clip(motor - result.command * 0.1265822784810128, 0, 1))
    assert result.command == 0
    assert abs(motor - 0.77) < 0.07
    assert not result.closing_episode


def test_close_threshold_preserves_loaded_width_and_closed_motor_goal():
    adapter = GripperPositionHold(1.0)
    first = update(adapter, 0.6, motor=0.65)
    assert first.command == 1 and first.closing_episode
    loaded = update(adapter, 0.529, measured=0.529, motor=0.47, previous=1.0)
    assert loaded.command == 1
    held = update(adapter, 0.529, measured=0.529, motor=0.0, previous=1.0)
    assert held.command == 0 and held.closing_episode and held.holding_width == 0.529


def test_legal_partial_open_release_does_not_immediately_relatch_below_close_threshold():
    adapter = GripperPositionHold(1.0)
    update(adapter, 0.05, measured=0.055, motor=0.0)
    pending = update(adapter, 0.117, measured=0.055, motor=0.0)
    assert pending.closing_episode
    released = update(adapter, 0.123, measured=0.055, motor=0.0)
    assert not released.closing_episode and released.command == -1
    rising = update(adapter, 0.13, measured=0.06, motor=0.12658, previous=-1.0)
    assert not rising.closing_episode and rising.command == 0
    # A new decline, rather than merely being below .60, permits a new latch.
    falling = update(adapter, 0.09, measured=0.06, motor=0.12658)
    assert falling.closing_episode and falling.command == 1


def test_release_requires_current_measured_clearance_and_full_observed_interval():
    adapter = GripperPositionHold(1.0)
    update(adapter, 0.529, measured=0.529, motor=0.0)
    assert update(adapter, 0.55, measured=0.529, motor=0.0).closing_episode
    assert update(adapter, 0.59, measured=0.529, motor=0.0).closing_episode
    released = update(adapter, 0.60, measured=0.529, motor=0.0)
    assert not released.closing_episode and released.command == -1


def test_invalid_long_interval_resets_release_evidence_without_retroactive_time_credit():
    adapter = GripperPositionHold(1.0)
    update(adapter, 0.53, motor=0.0)
    update(adapter, 0.61, motor=0.0)
    invalid = update(adapter, 0.62, motor=0.0, dt=0.5)
    assert not invalid.evidence_interval_valid and invalid.closing_episode
    assert update(adapter, 0.63, motor=0.0).closing_episode
    assert not update(adapter, 0.64, motor=0.0).closing_episode


def test_actual_previous_open_command_resets_release_evidence():
    adapter = GripperPositionHold(1.0)
    update(adapter, 0.53, motor=0.0)
    update(adapter, 0.61, motor=0.0)
    reset = update(adapter, 0.62, motor=0.0, previous=-1.0)
    assert reset.closing_episode and reset.release_evidence_s == 0


def test_no_closing_latch_when_low_reference_is_rising_from_initial_aperture():
    adapter = GripperPositionHold(0.1)
    result = update(adapter, 0.2, measured=0.1, motor=0.1)
    assert not result.closing_episode and result.command == -1


def test_episode_reset_and_instance_isolation():
    first, second = GripperPositionHold(1.0), GripperPositionHold(1.0)
    update(first, 0.53, motor=0.0)
    assert update(first, 0.53, motor=0.0).closing_episode
    assert not update(second, 0.95, motor=1.0).closing_episode
    first.reset(1.0)
    assert not update(first, 0.95, motor=1.0).closing_episode


@pytest.mark.parametrize(
    "kwargs",
    [{"close_threshold": True}, {"close_threshold": float("nan")}, {"motor_deadband": 0}, {"motor_deadband": 0.5}],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        GripperPositionHoldConfig(**kwargs)


@pytest.mark.parametrize("value", [True, float("nan"), -0.1, 1.1])
def test_invalid_online_opening_rejected(value):
    with pytest.raises(ValueError):
        update(GripperPositionHold(1.0), value)


def test_online_interface_contains_no_privileged_task_or_contact_argument():
    assert set(inspect.signature(GripperPositionHold.update).parameters) == {
        "self",
        "desired_opening",
        "measured_opening",
        "motor_opening",
        "previous_command",
        "dt",
    }
