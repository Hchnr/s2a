import numpy as np
import pytest

from s2a.qualified_enumerated_inverse import affine_response, inverse_translation


def test_actual_asymmetric_interface_and_bounded_solution_reprediction():
    rng = np.random.default_rng(42)
    jacobian = rng.normal(size=(12, 3))
    offset = rng.normal(size=12) * 0.001
    data = dict(
        action_command=np.zeros((1, 7)), action_goal_delta=np.zeros((1, 6)), current=np.array([[1.0, 2.0, 3.0]])
    )
    original = {k: v.copy() for k, v in data.items()}
    lo = np.array([-0.8, -1.0, -0.6])
    hi = np.array([0.9, 0.7, 1.0])
    outlo = np.array([-0.04, -0.06, -0.02])
    outhi = np.array([0.05, 0.03, 0.07])

    def predictor(row, _):
        assert np.all(row["action_command"][0, :3] >= lo) and np.all(row["action_command"][0, :3] <= hi)
        return (offset + jacobian @ row["action_goal_delta"][0, :3])[None, :]

    bounds = dict(input_min=lo, input_max=hi, output_min=outlo, output_max=outhi)
    response = affine_response(data, {}, predictor=predictor, **bounds)
    np.testing.assert_allclose(response["offset"], offset, atol=1e-14)
    np.testing.assert_allclose(response["jacobian"], jacobian, atol=1e-14)
    solved = inverse_translation(
        data, {}, [0.002, -0.001, 0.003], [0.0, 0.0, 0.0], previous_command=np.zeros(3), predictor=predictor, **bounds
    )
    assert np.max(np.abs(solved["command"])) <= 0.25 + 1e-12
    np.testing.assert_allclose(
        solved["full_predicted_response"], offset + jacobian @ solved["physical_delta"], atol=1e-14
    )
    for k in data:
        np.testing.assert_array_equal(data[k], original[k])

    def nonlinear(row, _):
        value = predictor(row, _)
        value[0, 0] += row["action_goal_delta"][0, 0] ** 2
        return value

    with pytest.raises(ValueError, match="not affine"):
        inverse_translation(
            data,
            {},
            [0.002, -0.001, 0.003],
            [0.0, 0.0, 0.0],
            previous_command=np.zeros(3),
            predictor=nonlinear,
            **bounds,
        )
