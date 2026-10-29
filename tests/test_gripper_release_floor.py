"""Noise suppression must retain a loaded motor but permit deliberate release."""

from s2a.gripper_position_hold import GripperPositionHold as Original
from s2a.gripper_release_floor import GripperPositionHold


def update(control, reference):
    return control.update(reference, 0.05, motor_opening=0.0, previous_command=0.0, dt=0.05)


def close(control):
    return update(control, 0.03)


def test_small_reference_rise_does_not_release_closed_motor():
    original, new = Original(0.8), GripperPositionHold(0.8)
    assert close(original).closing_episode and close(new).closing_episode
    for _ in range(3):
        old_result = update(original, 0.11)
        new_result = update(new, 0.11)
    assert not old_result.closing_episode
    assert new_result.closing_episode and new_result.command == 0
    assert new_result.reference == 0.11
    assert new_result.release_level == 0.35


def test_deliberate_release_and_independent_reset_remain_supported():
    control = GripperPositionHold(0.8)
    assert close(control).closing_episode
    update(control, 0.8)
    result = update(control, 0.8)
    assert not result.closing_episode and result.command == -1
    control.reset(0.8)
    result = update(control, 0.8)
    assert not result.closing_episode and result.release_evidence_s == 0
