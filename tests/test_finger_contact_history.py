import numpy as np
import pytest

from s2a.finger_contact_history import contact_history


def observations():
    return dict(
        x_finger_qpos=np.arange(12, dtype=float).reshape(6, 2) * 0.001,
        x_finger_qvel=np.arange(12, dtype=float).reshape(6, 2) * 0.01,
        x_motor_before=np.zeros((6, 2)),
        x_current_wrench_world=np.arange(36, dtype=float).reshape(6, 6),
        x_previous_command_available=np.array([0, 1, 1, 0, 1, 1]),
    )


def test_future_mutations_do_not_change_available_history_and_prefix_matches():
    data = observations()
    full = contact_history(data)
    prefix = {k: v[:3].copy() for k, v in data.items()}
    np.testing.assert_array_equal(contact_history(prefix), full[:3])
    changed = {k: v.copy() for k, v in data.items()}
    for key in changed:
        if key != "x_previous_command_available":
            changed[key][3:] += 100
    np.testing.assert_array_equal(contact_history(changed)[:3], full[:3])


def test_episode_reset_masks_old_measurements_and_differences_use_measured_values():
    data = observations()
    value = contact_history(data)
    np.testing.assert_array_equal(value[3], np.zeros(39))
    np.testing.assert_array_equal(value[4, 13:], np.zeros(26))
    np.testing.assert_allclose(value[4, :2], data["x_finger_qpos"][4] - data["x_finger_qpos"][3])
    np.testing.assert_allclose(value[4, 4:10], data["x_current_wrench_world"][4] - data["x_current_wrench_world"][3])
    assert value[4, 12] == 1
    # Labels and speculative reference packets have no effect on the feature.
    data["future_label"] = np.full((6, 100), np.nan)
    np.testing.assert_array_equal(contact_history(data), value)


def test_isolated_anchor_without_prefix_and_bad_observation_rejected():
    data = observations()
    data["x_previous_command_available"][0] = 1
    with pytest.raises(ValueError, match="episode start"):
        contact_history(data)
    data = observations()
    data["x_finger_qpos"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        contact_history(data)
