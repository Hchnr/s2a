import numpy as np
import pytest
from scipy.linalg import expm

from s2a.geometry import advance
from s2a.osc import OSCInterface, OSCStateToAction


def interface():
    return OSCInterface(-np.ones(6), np.ones(6), -np.array([0.05] * 3 + [0.5] * 3), np.array([0.05] * 3 + [0.5] * 3))


def test_fixed_pd_inverse_hits_endpoint_with_existing_velocity():
    kp, kd, dt = np.full(6, 150.0), np.full(6, 2 * np.sqrt(150)), 0.05
    converter = OSCStateToAction(interface(), kp, kd, dt, compensate=True)
    velocity = np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06])
    target_motion = np.array([0.002, -0.003, 0.004, 0.007, -0.008, 0.009])
    result = converter.convert(np.eye(4), advance(np.eye(4), target_motion, 1), velocity)
    # Independent augmented-state integration of the fixed second-order plant.
    for i in range(6):
        system = np.array([[0, 1, 0], [-kp[i], -kd[i], kp[i] * result["sent_delta"][i]], [0, 0, 0]])
        actual = (expm(system * dt) @ [0, velocity[i], 1])[0]
        np.testing.assert_allclose(actual, target_motion[i], atol=1e-12)
    assert not result["saturated"]


def test_baseline_matches_historical_affine_clip_and_inverse_is_bounded():
    converter = OSCStateToAction(interface(), np.full(6, 150), np.full(6, 25), 0.05)
    target = advance(np.eye(4), [0.08, -0.01, 0, 0, 0, 0.1], 1)
    baseline = converter.convert(np.eye(4), target, np.zeros(6))
    np.testing.assert_allclose(baseline["command"], [1, -0.2, 0, 0, 0, 0.2], atol=1e-14)
    compensated = OSCStateToAction(interface(), np.full(6, 150), np.full(6, 25), 0.05, compensate=True)
    result = compensated.convert(np.eye(4), target, np.zeros(6))
    assert result["saturated"]
    assert np.all(np.abs(result["command"]) <= 1)


def test_inverse_rejects_unobservable_interval_and_invalid_ranges():
    with pytest.raises(ValueError, match="insufficient"):
        OSCStateToAction(interface(), np.ones(6), np.ones(6), 1e-5)
    with pytest.raises(ValueError, match="increasing"):
        OSCInterface(np.ones(6), -np.ones(6), -np.ones(6), np.ones(6))
