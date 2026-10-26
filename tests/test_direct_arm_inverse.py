import numpy as np
from test_direct_arm_inverse_inputs import prepared

from s2a.direct_arm_inverse import command_prediction, nominal_translation, physical_parameters


def parameters():
    contract = dict(
        control_freq=20,
        control_frame="MuJoCo world",
        kp=np.ones(6) * 150,
        kd=np.ones(6) * 25,
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-np.ones(6) * 0.05,
        output_max=np.ones(6) * 0.05,
    )
    p = physical_parameters(contract)
    p.update(
        normalization_mean=np.zeros(89),
        normalization_scale=np.ones(89),
        frequencies=np.random.default_rng(42).normal(size=(64, 89)) / np.sqrt(89),
        coefficients=np.zeros((218, 3)),
    )
    return p


def test_physical_inverse_recovers_known_command_in_correct_units():
    data, target = prepared()
    p = parameters()
    wanted = np.array([0.2, -0.3, 0.1])
    delta = 0.05 * wanted
    target[:, :3, 3] = (
        np.asarray(p["translation_response"]) * delta
        + np.asarray(p["translation_momentum"]) * data["x_site_twist"][:, :3]
    )
    np.testing.assert_allclose(nominal_translation(data, target, p), [wanted, wanted], atol=1e-14)
    result = command_prediction(data, target, gripper_command=np.zeros(2), parameters=p)
    np.testing.assert_allclose(result["command"], [wanted, wanted], atol=1e-14)


def test_residual_prediction_is_bounded_and_never_uses_teacher_arm_labels():
    data, target = prepared()
    p = parameters()
    p["coefficients"][0] = [10, -10, 0]
    first = command_prediction(data, target, gripper_command=np.zeros(2), parameters=p)
    assert np.all(first["command"][:, :2] == [1, -1])
    data["action_command"][:] = np.nan
    data["action_goal_delta"][:] = np.nan
    second = command_prediction(data, target, gripper_command=np.zeros(2), parameters=p)
    np.testing.assert_array_equal(second["command"], first["command"])
