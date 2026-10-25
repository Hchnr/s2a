import numpy as np
import pytest

from s2a.residual_motor_inverse import ResidualMotorInverse


def parameters():
    return dict(
        target="next_motor_minus_current_motor",
        baseline="current_motor",
        mean=[0.0] * 7,
        scale=[1.0] * 7,
        coefficients=[0.0] * 40,
    )


def test_zero_residual_preserves_actual_target_despite_aperture_mismatch():
    model = ResidualMotorInverse(parameters())
    result = model.update(0.9, [-0.5, 0.5], [0.1] * 40, previous_opening=None)
    assert result.command == 0 and result.predicted_motor_target == -0.5
    assert result.selected_motor_target == -0.5 and result.previous_opening_delta == 0


def test_history_survives_chunk_boundaries_and_instances_do_not_share_state():
    p = parameters()
    p["coefficients"][2] = 10
    model = ResidualMotorInverse(p)
    assert model.update(0.6, [0], [0.6], previous_opening=0.5).command == -1
    assert model.update(0.6, [0], [0.6], previous_opening=None).command == 0
    assert ResidualMotorInverse(p).update(0.6, [0], [0.6], previous_opening=0.7).command == 1


def test_unavailable_reference_mask_and_no_access_beyond_eighth_endpoint():
    model = ResidualMotorInverse(parameters())
    assert model.update(0.5, [0], [0.5], previous_opening=None).available == (True, False, False, False)
    assert model.update(0.5, [0], [0.5] * 8 + [0.9] * 32, previous_opening=0.4) == model.update(
        0.5, [0], [0.5] * 8 + [0.1] * 32, previous_opening=0.4
    )


@pytest.mark.parametrize("cache", [[np.nan], [0, 0.1], [1.1], [], [0, 0, 0]])
def test_invalid_actual_cache_rejected(cache):
    with pytest.raises(ValueError):
        ResidualMotorInverse(parameters()).update(0.5, cache, [0.5], previous_opening=None)


def test_bad_parameters_rejected_before_execution():
    p = parameters()
    p["scale"][0] = 0
    with pytest.raises(ValueError):
        ResidualMotorInverse(p)
