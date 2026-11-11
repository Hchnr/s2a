"""Reset must not invent a completed plan or leak state between episodes."""

import numpy as np
import pytest

from s2a.research.joint_control.preview.reset_history import make_reset_preview_history


def staged_refiner(_inner):
    def refine(_env, _dynamics, _site, _twist, _poses, _anchor, _index, seed, _previous, _contract, history, **_kw):
        history.stage_plan(seed)
        return {"command": seed[0], "planned_commands": seed, "nominal_response": np.zeros(12)}

    return refine


def test_reset_binds_actual_initial_state_and_requires_completed_plan():
    site, twist = np.eye(4), np.zeros(6)
    reset = make_reset_preview_history(site, twist, refiner_factory=staged_refiner)
    np.testing.assert_array_equal(reset.previous_command, np.zeros(7))
    assert not reset.previous_command.flags.writeable
    assert reset.coupled.completed_periods == reset.tracking.history.completed_periods == 0
    assert reset.tracking.history.last_plan is None and reset.tracking.history.shifted(8) is None
    assert reset.inner_error.snapshot()["completed_periods"] == 0
    # Caller mutations cannot alter the bound measurement.
    site[0, 3] = 0.1
    with pytest.raises(ValueError, match="last completed"):
        reset.coupled.current(site, twist)
    with pytest.raises(ValueError, match="last completed"):
        reset.tracking.history.current(site, twist)
    np.testing.assert_array_equal(reset.coupled.current(np.eye(4), twist), np.zeros(12))


def test_only_observed_actions_create_a_warm_start_and_episodes_are_isolated():
    first = make_reset_preview_history(np.eye(4), np.zeros(6), refiner_factory=staged_refiner)
    second = make_reset_preview_history(np.eye(4), np.zeros(6), refiner_factory=staged_refiner)
    tx = first.tracking
    plan = np.zeros((8, 6))
    plan[1:, 0] = 0.1
    tx.prepare(
        None,
        {},
        np.eye(4),
        np.zeros(6),
        np.tile(np.eye(4), (8, 1, 1)),
        np.eye(4),
        0,
        plan,
        np.zeros(6),
        {},
        site_from_tcp=np.eye(4),
    )
    assert tx.history.last_plan is None
    tx.commit(plan[0])
    assert tx.history.last_plan is None
    with pytest.raises(ValueError, match="Observe"):
        tx.prepare(
            None,
            {},
            np.eye(4),
            np.zeros(6),
            np.tile(np.eye(4), (8, 1, 1)),
            np.eye(4),
            0,
            plan,
            np.zeros(6),
            {},
            site_from_tcp=np.eye(4),
        )
    tx.observe(np.eye(4), np.zeros(6))
    assert tx.history.completed_periods == 1
    np.testing.assert_array_equal(tx.history.shifted(8)[0], plan[1])
    assert second.tracking.history.completed_periods == 0 and second.tracking.history.last_plan is None
    assert second.inner_error.snapshot()["completed_periods"] == 0


def test_reset_rejects_invalid_pose_and_refiner():
    bad = np.eye(4)
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="rigid transform"):
        make_reset_preview_history(bad, np.zeros(6), refiner_factory=staged_refiner)
    with pytest.raises(TypeError, match="callable"):
        make_reset_preview_history(np.eye(4), np.zeros(6), refiner_factory=lambda _history: None)
