"""CPU controller qualification with a causal two-finger toy response.

The motor integrator follows the audited Panda native interface. The first-order
joint response/width stop is deliberately a test plant, not MuJoCo evidence.
"""

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from s2a.gripper import OpeningIntent, opening_position_target
from s2a.gripper_tracking import (
    GripperTrackingAdapter,
    GripperTrackingConfig,
    panda_opening_from_motor_target,
    panda_opening_from_qpos,
)


class JointResponse:
    def __init__(self, opening, *, minimum_gap=0.0, response_time_s=0.08):
        gap = 0.001 + 0.079 * opening
        self.q = np.array([gap / 2, -gap / 2])
        self.motor = opening_position_target(opening)
        self.minimum_gap = minimum_gap
        self.response_time_s = response_time_s

    @property
    def measured(self):
        return panda_opening_from_qpos(self.q)

    @property
    def target(self):
        return panda_opening_from_motor_target(self.motor)

    def advance(self, native_command):
        for _ in range(25):  # 20 Hz command; 0.002 s native control update.
            self.motor = np.clip(self.motor + np.array([-1, 1]) * 0.01 * np.sign(native_command), -1, 1)
            target_q = np.array([0.02, -0.02]) + 0.02 * self.motor
            self.q += (1 - np.exp(-0.002 / self.response_time_s)) * (target_q - self.q)
            if self.q[0] - self.q[1] < self.minimum_gap:
                self.q[:] = [self.minimum_gap / 2, -self.minimum_gap / 2]


def rollout(references, *, opening=1.0, minimum_gap=0.0, config=None):
    plant = JointResponse(opening, minimum_gap=minimum_gap)
    adapter = GripperTrackingAdapter(plant.measured, config=config)
    previous = 0.0
    results = []
    for reference in references:
        result = adapter.update(
            reference, plant.measured, motor_opening=plant.target, previous_command=previous, dt=0.05
        )
        plant.advance(result.command)
        results.append((result, plant.measured, plant.target))
        previous = result.command
    return adapter, plant, results


def begin_stalled_close():
    c = GripperTrackingConfig(reference_time_constant_s=0)
    adapter = GripperTrackingAdapter(0.65, config=c)
    first = adapter.update(0.2, 0.65, motor_opening=0.9, previous_command=0, dt=0.05)
    assert first.command == 1
    previous = 1
    for _ in range(8):
        result = adapter.update(0.2, 0.65, motor_opening=0.15, previous_command=previous, dt=0.05)
        previous = result.command
    assert result.blocked_candidate and result.command == 0
    return adapter


def test_panda_joint_and_motor_contract_round_trip_and_native_zero_hold():
    for opening in [0, 0.2, 0.7, 1]:
        gap = 0.001 + 0.079 * opening
        assert panda_opening_from_qpos([gap / 2, -gap / 2]) == pytest.approx(opening)
        assert panda_opening_from_motor_target(opening_position_target(opening)) == pytest.approx(opening)
    plant = JointResponse(0.8, minimum_gap=0.04)
    for _ in range(8):
        plant.advance(1)
    motor_before = plant.motor.copy()
    assert plant.measured > plant.target + 0.1
    plant.advance(0)
    np.testing.assert_array_equal(plant.motor, motor_before)
    assert plant.measured > plant.target + 0.1  # Closing target offset survives a zero command.


def test_free_joint_endpoint_closing_and_reopening_without_contact_input():
    _, plant, rows = rollout([1] * 20 + [0] * 60 + [1] * 60)
    assert rows[79][1] < 0.03
    assert plant.measured > 0.96
    assert any(r.command == 1 for r, _, _ in rows)
    assert any(r.command == -1 for r, _, _ in rows[80:])
    assert all(r.command in (-1, 0, 1) for r, _, _ in rows)
    assert not any(r.blocked_candidate for r, _, _ in rows)  # Slow response alone below these settings did not qualify.


def test_reference_noise_does_not_make_sign_controller_reverse_at_rest():
    refs = np.tile([0.58, 0.62], 100)
    adapter = GripperTrackingAdapter(0.6)
    intent = OpeningIntent(0.6)
    historical_sign = [intent.update(float(x)) for x in refs]
    proposed = [adapter.update(float(x), 0.6, motor_opening=0.6, previous_command=0, dt=0.05).command for x in refs]
    assert np.count_nonzero(np.diff(historical_sign)) > 100
    assert proposed == [0] * len(refs)


def test_low_reference_rise_does_not_release_below_current_blocked_width():
    adapter = begin_stalled_close()
    # A large rise from the old minimum is insufficient while still below
    # current aperture plus clearance. No scene label or contact is supplied.
    for reference in [0.4, 0.55, 0.64, 0.67, 0.69] * 4:
        result = adapter.update(reference, 0.65, motor_opening=0.15, previous_command=0, dt=0.05)
        assert result.command == 0 and result.closing_episode
        assert result.release_evidence_s == 0
    assert result.release_level == pytest.approx(0.71)


def test_release_requires_contiguous_time_and_resets_on_noise_or_observation_gap():
    adapter = begin_stalled_close()

    def step(reference, dt=0.05):
        return adapter.update(reference, 0.65, motor_opening=0.15, previous_command=0, dt=dt)

    assert step(0.9).command == 0
    assert step(0.9).release_evidence_s == pytest.approx(0.05)
    assert step(0.65).release_evidence_s == 0
    gap = step(0.9, dt=0.5)
    assert not gap.evidence_interval_valid and gap.command == 0 and gap.release_evidence_s == 0
    results = [step(0.9) for _ in range(4)]
    assert [r.command for r in results] == [0, 0, 0, -1]
    assert results[-1].reason == "measured_clearance_and_reference_rise_confirmed"


def test_blocked_width_stop_retains_compression_and_later_opens_in_toy_plant():
    gap = 0.001 + 0.079 * 0.55
    _, plant, rows = rollout([0.1] * 80 + [0.95] * 80, minimum_gap=gap)
    blocked_rows = [r for r, _, _ in rows[:80] if r.blocked_candidate]
    assert blocked_rows and all(r.command == 0 for r in blocked_rows)
    assert rows[79][1] == pytest.approx(0.55)
    assert rows[79][2] < rows[79][1] - 0.1
    assert plant.measured > 0.85
    assert any(r.command == -1 for r, _, _ in rows[80:])


def test_full_open_request_can_unload_a_wide_stall_without_waiting_for_ema_to_equal_one():
    adapter = GripperTrackingAdapter(1.0)
    previous = 0
    for _ in range(20):
        result = adapter.update(0.6, 1.0, motor_opening=0.7, previous_command=previous, dt=0.05)
        previous = result.command
    assert result.closing_episode and result.blocked_candidate
    results = []
    for _ in range(20):
        result = adapter.update(1.0, 1.0, motor_opening=0.7, previous_command=0, dt=0.05)
        results.append(result)
        if result.command == -1:
            break
    assert results[-1].command == -1
    assert results[-1].filtered_reference < 1


def test_stall_candidate_can_be_a_stuck_motor_and_is_invalidated_by_actual_open_override():
    # Identical available measurements can result from obstruction or a stuck
    # joint. The state deliberately says candidate, without claiming grasp.
    adapter = begin_stalled_close()
    result = adapter.update(0.2, 0.65, motor_opening=0.2, previous_command=-1, dt=0.05)
    assert not result.blocked_candidate and not result.closing_episode
    assert "contact" not in asdict(result)


def test_achieved_motor_target_stops_driving_before_lagging_measured_aperture():
    adapter = GripperTrackingAdapter(1.0, config=GripperTrackingConfig(reference_time_constant_s=0))
    first = adapter.update(0.5, 1.0, motor_opening=1, previous_command=0, dt=0.05)
    assert first.command == 1
    later = adapter.update(0.5, 0.8, motor_opening=0.4, previous_command=1, dt=0.05)
    assert later.command == 0
    assert later.measured_opening > later.filtered_reference


def test_closing_preload_is_not_cancelled_by_motor_deadband():
    adapter = GripperTrackingAdapter(0.5, config=GripperTrackingConfig(reference_time_constant_s=0))
    adapter.update(0.3, 0.5, motor_opening=0.8, previous_command=0, dt=0.05)
    result = adapter.update(0.5, 0.5, motor_opening=0.5, previous_command=1, dt=0.05)
    assert result.closing_episode and result.command == 1


def test_same_available_prefix_is_identical_despite_different_future_references():
    prefix = [1] * 10 + [0.3] * 30
    _, _, first = rollout(prefix + [1] * 40)
    _, _, second = rollout(prefix + [0] * 40)
    assert first[: len(prefix)] == second[: len(prefix)]
    assert [r.command for r, _, _ in first[len(prefix) :]] != [r.command for r, _, _ in second[len(prefix) :]]


def test_reset_isolates_rollouts_and_does_not_reset_at_chunk_boundaries():
    adapter = begin_stalled_close()
    assert adapter.update(0.3, 0.65, motor_opening=0.15, previous_command=0, dt=0.05).blocked_candidate
    adapter.reset(0.8)
    fresh = GripperTrackingAdapter(0.8, config=adapter.config)
    for reference in [0.8, 0.82, 0.7, 0.6]:
        args = dict(motor_opening=0.8, previous_command=0, dt=0.05)
        assert adapter.update(reference, 0.8, **args) == fresh.update(reference, 0.8, **args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("desired_opening", np.nan),
        ("measured_opening", np.inf),
        ("motor_opening", -0.01),
        ("previous_command", 2),
        ("dt", 0),
        ("dt", -0.1),
        ("desired_opening", True),
        ("motor_opening", [0.2]),
    ],
)
def test_rejected_sample_does_not_change_filter_or_history(field, value):
    adapter = begin_stalled_close()
    before = deepcopy(vars(adapter))
    args = dict(desired_opening=0.8, measured_opening=0.65, motor_opening=0.15, previous_command=0, dt=0.05)
    args[field] = value
    with pytest.raises(ValueError):
        adapter.update(**args)
    assert vars(adapter) == before


def test_invalid_configuration_and_motor_feedback_are_rejected():
    for field, value in [
        ("release_margin", 0.01),
        ("release_confirmation_s", 0),
        ("reference_time_constant_s", -1),
        ("stall_speed_per_s", np.nan),
        ("closing_preload", 1),
        ("max_evidence_interval_s", True),
    ]:
        with pytest.raises(ValueError):
            replace(GripperTrackingConfig(), **{field: value})
    for bad in [[0], [0, 1, 2], [np.nan, 0], [1.1, -1.1]]:
        with pytest.raises(ValueError):
            panda_opening_from_motor_target(bad)
