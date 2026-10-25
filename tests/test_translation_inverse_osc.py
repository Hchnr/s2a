import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from s2a.osc import OSCInterface, OSCStateToAction
from s2a.translation_inverse_osc import TranslationInverseOSC


def interface():
    return OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))


def test_full_translation_hits_nominal_endpoint_with_current_velocity():
    kp = np.full(6, 150.0)
    kd = 2 * np.sqrt(kp)
    dt = 0.05
    model = TranslationInverseOSC(interface(), kp, kd, dt)
    desired = np.eye(4)
    desired[:3, 3] = [0.002, -0.003, 0.001]
    velocity = np.array([0.01, -0.02, 0.005, 0, 0, 0])
    result = model.convert(np.eye(4), desired, velocity)
    a = expm(np.array([[0, 1], [-150.0, -2 * np.sqrt(150.0)]]) * dt)
    actual_nominal = (1 - a[0, 0]) * result["sent_delta"][:3] + a[0, 1] * velocity[:3]
    np.testing.assert_allclose(actual_nominal, desired[:3, 3], rtol=0, atol=1e-12)
    assert not result["saturated"] and not result["rate_limited"]


@pytest.mark.parametrize("angle", [0.1, 0.4])
def test_rotational_behavior_is_identical_including_large_rotation_guard(angle):
    kp = np.full(6, 150.0)
    kd = 2 * np.sqrt(kp)
    desired = np.eye(4)
    desired[:3, :3] = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
    velocity = np.array([0, 0, 0, 0.02, -0.01, 0.03])
    old = OSCStateToAction(interface(), kp, kd, 0.05, compensate=True, inverse_weight=0.5).convert(
        np.eye(4), desired, velocity
    )
    new = TranslationInverseOSC(interface(), kp, kd, 0.05).convert(np.eye(4), desired, velocity)
    np.testing.assert_array_equal(old["command"][3:], new["command"][3:])
    assert old["rotation_inverse_enabled"] == new["rotation_inverse_enabled"]


def test_rate_state_and_input_saturation_are_retained():
    kp = np.full(6, 150.0)
    model = TranslationInverseOSC(interface(), kp, 2 * np.sqrt(kp), 0.05, max_command_rate=5)
    desired = np.eye(4)
    desired[0, 3] = 10
    first = model.convert(np.eye(4), desired, np.zeros(6))
    second = model.convert(np.eye(4), desired, np.zeros(6))
    assert first["saturated"] and first["rate_limited"]
    assert first["command"][0] == 0.25 and second["command"][0] == 0.5
    fresh = TranslationInverseOSC(interface(), kp, 2 * np.sqrt(kp), 0.05, max_command_rate=5)
    assert fresh.convert(np.eye(4), desired, np.zeros(6))["command"][0] == 0.25


def test_entire_conversion_diff_is_only_registered_translation_weights():
    old = Path("src/s2a/osc.py").read_text()
    new = Path("src/s2a/translation_inverse_osc.py").read_text()
    new = new.replace(
        "            weights = np.array([1.0, 1.0, 1.0, self.inverse_weight, self.inverse_weight, self.inverse_weight])\n            delta += weights * (inverse_delta - endpoint_error)",
        "            delta += self.inverse_weight * (inverse_delta - endpoint_error)",
    )

    def convert(text):
        return next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "convert")

    assert ast.dump(convert(old), include_attributes=False) == ast.dump(convert(new), include_attributes=False)
