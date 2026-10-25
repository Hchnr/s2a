import numpy as np
import pytest

from s2a.velocity_residual_response import VelocityResidualResponse, motor_after_native_period


def parameters():
    return dict(
        velocity_units="normalized_opening_per_20Hz_period",
        prior="constant_current_velocity",
        mean=[0.0] * 4,
        scale=[1.0] * 4,
        coefficients=np.zeros((15, 2)),
    )


def test_zero_residual_preserves_measured_velocity_instead_of_inferring_from_motor_gap():
    model = VelocityResidualResponse(parameters())
    result = model.predict(0.45, 1.0, 0.05, -1.0)
    assert result.next_opening == 0.5 and result.next_period_velocity == 0.05 and result.next_motor_target == 1
    assert model.predict(0.45, 1.0, 0.01, -1.0).next_opening == 0.46


def test_rollout_uses_its_own_state_and_does_not_clip_velocity():
    model = VelocityResidualResponse(parameters())
    result = model.predict(0.9, 0, 0.3, 0)
    assert result.next_opening == 1 and result.next_period_velocity == 0.3
    second = model.predict(result.next_opening, result.next_motor_target, result.next_period_velocity, 0)
    assert second.next_opening == 1 and second.next_period_velocity == 0.3


def test_two_outputs_and_parameter_ownership():
    p = parameters()
    p["coefficients"][-1] = [0.01, -0.02]
    model = VelocityResidualResponse(p)
    p["coefficients"][:] = 0
    result = model.predict(0.5, 0, 0.05, 1)
    assert result.next_opening == pytest.approx(0.56) and result.next_period_velocity == pytest.approx(0.03)
    assert result.next_motor_target == pytest.approx(-0.25)


def test_motor_update_matches_substeps_including_saturation():
    assert motor_after_native_period(0.95, -1) == 1
    assert motor_after_native_period(-0.95, 1) == -1
    assert motor_after_native_period(0.3, 0) == 0.3


def test_nonfinite_rollout_is_rejected_instead_of_hidden_by_opening_clip():
    p = parameters()
    p["coefficients"][:] = 1e308
    with pytest.raises(ValueError, match="Nonfinite"):
        VelocityResidualResponse(p).predict(0.5, 1, 1e300, -1)


@pytest.mark.parametrize("state", [(np.nan, 0, 0, 0), (0.5, 1.1, 0, 0), (0.5, 0, np.inf, 0), (0.5, 0, 0, 2)])
def test_invalid_current_state_or_command_fails(state):
    with pytest.raises(ValueError):
        VelocityResidualResponse(parameters()).predict(*state)
