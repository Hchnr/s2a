from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from s2a.gripper import OpeningIntent
from s2a.gripper_intent_hold import GripperIntentHold, GripperIntentHoldConfig


def update(adapter, reference, measured=0.6, motor=0.0, previous=0, dt=0.05):
    return adapter.update(reference, measured, motor_opening=motor, previous_command=previous, dt=dt)


def test_closing_onset_matches_old_intent_without_reference_filter_delay():
    refs = [0.85, 0.88, 0.87, 0.84, 0.7, 0.5]
    adapter = GripperIntentHold(0.85)
    old = OpeningIntent(0.85)
    expected = [old.update(x) for x in refs]
    actual = [update(adapter, x, measured=0.85, motor=0.85).command for x in refs]
    assert next(i for i, c in enumerate(actual) if c > 0) == next(i for i, c in enumerate(expected) if c > 0)


def test_observed_width_is_not_used_as_free_motor_closing_target():
    adapter = GripperIntentHold(0.9)
    assert update(adapter, 0.5, motor=0.9).command == 1
    for motor in [0.6, 0.5, 0.3, 0.1]:
        assert update(adapter, 0.5, measured=0.5, motor=motor, previous=1).command == 1
    held = update(adapter, 0.5, measured=0.5, motor=0.0, previous=1)
    assert held.command == 0 and held.state == "closed_motor_hold"
    assert held.holding_width == 0.5


def test_reference_rise_without_current_width_clearance_cannot_release():
    adapter = GripperIntentHold(0.9)
    update(adapter, 0.1, measured=0.7, motor=0.9)
    for reference in [0.2, 0.4, 0.68, 0.71, 0.72] * 3:
        r = update(adapter, reference, measured=0.7)
        assert r.command == 0 and r.closing_episode
    assert r.release_level == pytest.approx(0.74)


def test_sustained_clearance_releases_after_one_observed_interval_and_opens_to_limit():
    adapter = GripperIntentHold(0.9)
    update(adapter, 0.2, motor=0.9)
    first = update(adapter, 0.8)
    second = update(adapter, 0.8)
    assert first.command == 0 and first.release_evidence_s == 0
    assert second.command == -1 and second.release_evidence_s == pytest.approx(0.05)
    assert update(adapter, 0.8, measured=0.79, motor=0.8, previous=-1).command == -1
    assert update(adapter, 0.8, measured=1, motor=1, previous=-1).command == 0


def test_pending_release_does_not_reduce_closing_effort_before_motor_limit():
    adapter = GripperIntentHold(0.9)
    update(adapter, 0.2, motor=0.9)
    pending = update(adapter, 0.9, measured=0.6, motor=0.4, previous=1)
    assert pending.command == 1 and pending.release_evidence_s == 0


def test_spike_and_observation_gap_do_not_accumulate_release_time():
    adapter = GripperIntentHold(0.9)
    update(adapter, 0.2, motor=0.9)
    update(adapter, 0.9)
    assert update(adapter, 0.3).release_evidence_s == 0
    gap = update(adapter, 0.9, dt=0.5)
    assert not gap.evidence_interval_valid and gap.command == 0
    assert update(adapter, 0.9).command == 0
    assert update(adapter, 0.9).command == -1


def test_no_contact_claim_and_wide_aperture_can_still_receive_full_open_request():
    adapter = GripperIntentHold(1)
    update(adapter, 0.8, measured=1, motor=1)
    assert update(adapter, 1, measured=1, motor=0).command == 0
    assert update(adapter, 1, measured=1, motor=0).command == -1


def test_actual_open_override_invalidates_held_width_time_evidence():
    adapter = GripperIntentHold(0.9)
    update(adapter, 0.2, motor=0.9)
    update(adapter, 0.9)
    r = update(adapter, 0.9, motor=0.4, previous=-1)
    assert r.release_evidence_s == 0 and r.command == 1 and r.holding_width is None


def test_reset_and_rejected_sample_leave_rollouts_isolated():
    adapter = GripperIntentHold(0.9)
    update(adapter, 0.2, motor=0.9)
    before = deepcopy(vars(adapter))
    with pytest.raises(ValueError):
        update(adapter, np.nan)
    assert vars(adapter).keys() == before.keys()
    assert vars(adapter.intent) == vars(before["intent"])
    assert adapter._reference_min == before["_reference_min"]
    adapter.reset(0.7)
    fresh = GripperIntentHold(0.7)
    for x in [0.7, 0.6, 0.4, 0.8, 0.9]:
        assert update(adapter, x) == update(fresh, x)


def test_invalid_settings_rejected():
    for name, value in [
        ("release_margin", 0.01),
        ("motor_limit_tolerance", 1),
        ("release_confirmation_s", 0),
        ("release_excursion", np.inf),
        ("reference_hysteresis", True),
    ]:
        with pytest.raises(ValueError):
            replace(GripperIntentHoldConfig(), **{name: value})
