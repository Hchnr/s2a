import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from s2a.osc import OSCInterface, OSCStateToAction
from s2a.pose_velocity_inverse_osc import PoseVelocityInverseOSC
from s2a.reference_linear_velocity import reference_linear_velocity


def interface():
    return OSCInterface(-np.ones(6), np.ones(6), np.array([-0.05] * 3 + [-0.5] * 3), np.array([0.05] * 3 + [0.5] * 3))


def model(rate=None):
    kp = np.full(6, 150.0)
    return PoseVelocityInverseOSC(interface(), kp, 2 * np.sqrt(kp), 0.05, max_command_rate=rate)


def test_unconstrained_two_state_objective_has_zero_gradient_and_beats_pose_only_inverse():
    m = model()
    desired = np.eye(4)
    desired[:3, 3] = [0.002, -0.001, 0.003]
    velocity = np.array([0.01, -0.02, 0.005, 0, 0, 0])
    reference = np.array([0.04, -0.01, 0.02])
    got = m.convert(np.eye(4), desired, velocity, desired_linear_velocity=reference)
    a = expm(np.array([[0, 1], [-150.0, -2 * np.sqrt(150.0)]]) * 0.05)
    r, b, c, d = 1 - a[0, 0], a[0, 1], -a[1, 0], a[1, 1]
    delta = got["requested_delta"][:3]
    error = desired[:3, 3]
    gradient = r * (r * delta + b * velocity[:3] - error) + 0.05**2 * c * (c * delta + d * velocity[:3] - reference)
    np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-12)

    def loss(value):
        return np.sum(
            (r * value + b * velocity[:3] - error) ** 2 + 0.05**2 * (c * value + d * velocity[:3] - reference) ** 2
        )

    assert loss(delta) <= loss((error - b * velocity[:3]) / r)
    assert not got["saturated"] and not got["rate_limited"]


def test_box_and_rate_projection_is_constrained_objective_minimum():
    m = model(5)
    desired = np.eye(4)
    desired[0, 3] = 0.01
    v = np.array([0.02, 0, 0, 0, 0, 0])
    w = np.array([0.3, 0, 0])
    got = m.convert(np.eye(4), desired, v, desired_linear_velocity=w)
    assert got["saturated"] and got["rate_limited"] and got["command"][0] == 0.25
    a = expm(np.array([[0, 1], [-150.0, -2 * np.sqrt(150.0)]]) * 0.05)
    r, b, c, d = 1 - a[0, 0], a[0, 1], -a[1, 0], a[1, 1]

    def cost(u):
        delta = 0.05 * u
        return (r * delta + b * v[0] - 0.01) ** 2 + 0.05**2 * (c * delta + d * v[0] - 0.3) ** 2

    assert cost(0.25) <= min(cost(u) for u in np.linspace(-0.25, 0.25, 101)) + 1e-14


@pytest.mark.parametrize("angle", [0.1, 0.4])
def test_rotation_is_original_half_compensation(angle):
    desired = np.eye(4)
    desired[:3, :3] = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
    v = np.array([0, 0, 0, 0.02, 0.01, -0.01])
    kp = np.full(6, 150.0)
    old = OSCStateToAction(interface(), kp, 2 * np.sqrt(kp), 0.05, compensate=True, inverse_weight=0.5)
    baseline = old.convert(np.eye(4), desired, v)
    new = model().convert(np.eye(4), desired, v, desired_linear_velocity=np.zeros(3))
    np.testing.assert_array_equal(new["command"][3:], baseline["command"][3:])
    assert new["rotation_inverse_enabled"] == baseline["rotation_inverse_enabled"]


def poses():
    values = np.repeat(np.eye(4)[None], 4, axis=0)
    values[:, 0, 3] = [0.01, 0.03, 0.06, 0.1]
    return values


def test_reference_derivative_modes_and_generated_tail_causality():
    p = poses()
    anchor = np.eye(4)
    first = reference_linear_velocity(p, 0, anchor, 0.05)
    middle = reference_linear_velocity(p, 1, anchor, 0.05)
    last = reference_linear_velocity(p, 3, anchor, 0.05)
    assert (
        first["mode"] == "central_query_anchor"
        and first["before_index"] == -1
        and first["velocity"][0] == pytest.approx(0.3)
    )
    assert middle["mode"] == "central_generated" and middle["velocity"][0] == pytest.approx(0.5)
    assert last["mode"] == "backward_tail" and last["velocity"][0] == pytest.approx(0.8)
    one = reference_linear_velocity(p[:1], 0, anchor, 0.05)
    assert one["mode"] == "single_anchor" and one["velocity"][0] == pytest.approx(0.2)
    changed = p.copy()
    changed[2:, 0, 3] = 10
    np.testing.assert_array_equal(reference_linear_velocity(changed, 0, anchor, 0.05)["velocity"], first["velocity"])


@pytest.mark.parametrize("index,dt", [(True, 0.05), (-1, 0.05), (4, 0.05), (0, 0)])
def test_invalid_reference_interval_is_not_silently_repaired(index, dt):
    with pytest.raises(ValueError):
        reference_linear_velocity(poses(), index, np.eye(4), dt)
