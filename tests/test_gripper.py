import numpy as np
import pytest

from s2a.gripper import OpeningIntent, opening_position_target


def test_contact_plateau_preserves_closing_intent_then_releases():
    adapter = OpeningIntent(0.9, band=0.02)
    commands = [adapter.update(x) for x in [0.95, 1.0, 0.99, 0.96, 0.5, 0.51, 0.5, 0.53, 0.9]]
    assert commands == [-1, -1, -1, 1, 1, 1, 1, -1, -1]
    assert OpeningIntent(0.9).update(0.9) == -1


def test_hysteresis_tracks_cumulative_excursion_and_rejects_invalid_inputs():
    adapter = OpeningIntent(1.0, band=0.02)
    assert [adapter.update(x) for x in [0.995, 0.99, 0.985, 0.97]] == [-1, -1, -1, 1]
    for x in [np.nan, np.inf, -0.1, 1.1]:
        with pytest.raises(ValueError, match="opening"):
            adapter.update(x)
    with pytest.raises(ValueError, match="band"):
        OpeningIntent(0.5, band=0)


def test_position_bridge_uses_actuator_affine_mapping_not_native_sign():
    for opening in [0, 0.4, 1]:
        command = opening_position_target(opening)
        q = np.array([0.02, -0.02]) + 0.02 * command
        np.testing.assert_allclose(q[0] - q[1], 0.079 * opening + 0.001)
