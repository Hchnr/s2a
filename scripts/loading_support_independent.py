"""166 independently audit support choice, learned KKT and executed mixture."""

import numpy as np
from arm_joint_wrench_independent import conditions, prediction
from qualified_inverse_independent import verify_inverse


def verify_support_inverse(
    data,
    parameters,
    contract,
    previous,
    desired_motion,
    desired_velocity,
    physical_candidate,
    support_points,
    radius,
    logged,
    *,
    expected_route,
):
    z = (conditions(data)[0] - np.asarray(parameters["state_mean"])) / np.asarray(parameters["state_scale"])
    points = np.asarray(support_points, dtype=float)
    assert points.ndim == 2 and points.shape[1] == 89 and len(points)
    assert np.isfinite(points).all() and np.isfinite(radius) and radius > 0
    distances = np.sqrt(np.sum((points - z) ** 2, axis=1))
    distance = float(distances.min())
    route = logged["arm_loading_route"]
    assert route == expected_route
    assert route in ("native_close_load", "model_tracking")
    weight = 0.0 if route == "native_close_load" else 0.5 * min(1.0, max(0.0, 2.0 - distance / radius))
    decision = logged["support_decision"]
    np.testing.assert_allclose(decision["distance"], distance, atol=1e-10, rtol=0)
    index = int(decision["nearest_index"])
    assert 0 <= index < len(points)
    np.testing.assert_allclose(distances[index], distance, atol=1e-10, rtol=0)
    np.testing.assert_allclose(decision["learned_weight"], weight, atol=1e-12, rtol=0)
    learned = logged["learned_solution"]
    checked = verify_inverse(data, parameters, contract, previous, desired_motion, desired_velocity, learned["command"])
    np.testing.assert_allclose(learned["full_predicted_response"], checked["full_prediction"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(learned["squared_objective"], checked["objective"], atol=1e-12, rtol=0)
    physical_candidate = np.asarray(physical_candidate, dtype=float)
    assert physical_candidate.shape == (3,) and np.isfinite(physical_candidate).all()
    np.testing.assert_allclose(logged["physical_candidate"], physical_candidate, atol=1e-12, rtol=0)
    previous = np.asarray(previous, dtype=float)
    lo, hi, outlo, outhi = (np.asarray(contract[k])[:3] for k in ("input_min", "input_max", "output_min", "output_max"))
    assert previous.shape == (3,) and np.all(previous >= lo) and np.all(previous <= hi)
    mixture = weight * np.asarray(learned["command"]) + (1 - weight) * physical_candidate
    command = np.minimum(np.minimum(hi, previous + 0.25), np.maximum(np.maximum(lo, previous - 0.25), mixture))
    np.testing.assert_allclose(logged["command"], command, atol=1e-12, rtol=0)
    physical = outlo + (command - lo) * (outhi - outlo) / (hi - lo)
    actual = {k: np.asarray(v).copy() for k, v in data.items()}
    actual["action_command"][0, :3] = command
    actual["action_goal_delta"][0, :3] = physical
    full = prediction(actual, parameters)[0]
    np.testing.assert_allclose(logged["physical_delta"], physical, atol=1e-12, rtol=0)
    np.testing.assert_allclose(logged["full_predicted_response"], full, atol=1e-9, rtol=0)
    np.testing.assert_allclose(logged["predicted_motion"], full[:3], atol=1e-9, rtol=0)
    np.testing.assert_allclose(logged["predicted_linear_velocity"], full[6:9] / 0.05, atol=1e-9, rtol=0)
    residual = full[[0, 1, 2, 6, 7, 8]] - np.r_[desired_motion, 0.05 * np.asarray(desired_velocity)]
    np.testing.assert_allclose(logged["squared_objective"], residual @ residual, atol=1e-12, rtol=0)
    return dict(
        command=command,
        physical_delta=physical,
        full_prediction=full,
        objective=float(residual @ residual),
        learned_weight=weight,
    )
