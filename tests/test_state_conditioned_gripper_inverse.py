import numpy as np
import pytest

from s2a.state_conditioned_gripper_inverse import design_features, raw_features, reachable_command


def test_only_requested_generated_prefix_affects_features():
    a, mask = raw_features(0.4, -0.5, 0.01, np.linspace(0.2, 0.9, 10))
    altered = np.linspace(0.2, 0.9, 10)
    altered[8:] = 0
    b, other = raw_features(0.4, -0.5, 0.01, altered)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(mask, other)
    short, mask = raw_features(0.4, -0.5, 0.01, [0.2])
    np.testing.assert_array_equal(mask, [True, False, False, False])
    np.testing.assert_allclose(short[3:], [-0.2] * 4)


def test_polynomial_and_mask_dimensions_are_explicit():
    x, mask = raw_features(0.4, 0.0, 0.0, [0.5, 0.6])
    features = design_features(x, mask, np.zeros(7), np.ones(7))
    assert features.shape == (40,) and features[-1] == 1
    np.testing.assert_array_equal(features[-5:-1], mask.astype(float))
    with pytest.raises(ValueError):
        design_features(x, mask.astype(int), np.zeros(7), np.ones(7))


def test_command_choice_respects_increment_saturation_and_zero_tie():
    assert reachable_command(0.0, 0.25)["command"] == -1
    assert reachable_command(0.0, -0.25)["command"] == 1
    assert reachable_command(0.0, 0.125)["command"] == 0
    assert reachable_command(1.0, 3.0)["command"] == 0
    assert reachable_command(-1.0, -3.0)["command"] == 0
    assert reachable_command(0.9, 1.0)["selected_motor_target"] == 1


@pytest.mark.parametrize("reference", [[], [float("nan")], [1.2]])
def test_invalid_references_rejected(reference):
    with pytest.raises(ValueError):
        raw_features(0.4, 0, 0, reference)
