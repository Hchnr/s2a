"""Position bounds must remain conservative for box and rotational motion."""

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.research.joint_control.tracking.causal_position_reachability import causal_position_outer_bounds
from s2a.research.joint_control.tracking.stopped_pose_safety_bound import first_pose_linearization
from scripts.verification.reference_ports.check_stopped_pose_safety_bound import check_support_certificate


def test_position_bound_uses_the_completed_innovation_and_relaxed_box():
    linearization = {
        "target_position": np.array([0.008, 0, 0]),
        "seed_tcp_position": np.zeros(3),
        "position_jacobian": np.array([[0.01, 0, 0, 0, 0, 0], [0, 0.02, 0, 0, 0, 0], [0, 0, 0.01, 0, 0, 0]]),
        "position_remainder_bound_m": 0.0,
    }
    box = {
        "lower": -np.ones(6) * 0.1,
        "upper": np.ones(6) * 0.1,
        "a": np.empty((0, 6)),
        "b": np.empty(0),
        "seed": np.zeros(6),
    }
    result = causal_position_outer_bounds(linearization, [0.002, 0, 0], {"box": box})
    bound = result["scopes"]["box"]
    np.testing.assert_allclose(bound["position_lower_bound_m"], 0.009 - 1e-9, atol=1e-12, rtol=0)
    check_support_certificate(bound["support_direction"], box, bound["support"])
    rng = np.random.default_rng(42)
    for command in rng.uniform(box["lower"], box["upper"], size=(128, 6)):
        error = np.array([0.01, 0, 0]) - linearization["position_jacobian"] @ command
        assert np.linalg.norm(error) >= bound["position_lower_bound_m"]


def test_exponential_remainder_keeps_bound_below_sampled_nonlinear_tcp_errors():
    site, tool, target = np.eye(4), np.eye(4), np.eye(4)
    tool[:3, 3] = [0.06, -0.03, 0.14]
    target[:3, 3] = [0.09, -0.01, 0.16]
    jacobian = np.c_[np.eye(6), np.zeros(6)]
    command_jacobian = np.vstack((np.eye(6) * 0.1, np.zeros(6)))
    seed, lower, upper = np.zeros(6), -np.ones(6) * 0.2, np.ones(6) * 0.2
    linearization = first_pose_linearization(
        site, tool, jacobian, np.zeros(7), command_jacobian, seed, lower, upper, target
    )
    box = {"lower": lower, "upper": upper, "a": np.empty((0, 6)), "b": np.empty(0), "seed": seed}
    innovation = np.array([0.001, -0.002, 0.003])
    bound = causal_position_outer_bounds(linearization, innovation, {"box": box})["scopes"]["box"][
        "position_lower_bound_m"
    ]
    rng = np.random.default_rng(42)
    for command in rng.uniform(lower, upper, size=(512, 6)):
        movement = jacobian @ command_jacobian @ command
        tcp = movement[:3] + Rotation.from_rotvec(movement[3:]).apply(tool[:3, 3])
        assert np.linalg.norm(target[:3, 3] - tcp + innovation) >= bound - 1e-12


def test_linear_constraint_support_is_independently_reconstructible():
    linearization = {
        "target_position": np.array([0.01, 0, 0]),
        "seed_tcp_position": np.zeros(3),
        "position_jacobian": np.c_[np.diag([0.01, 0.02, 0.01]), np.zeros((3, 3))],
        "position_remainder_bound_m": 0.0,
    }
    row = np.array([[1.0, 0, 0, 0, 0, 0]])
    scope = {"lower": -np.ones(6), "upper": np.ones(6), "a": row, "b": np.array([0.05]), "seed": np.zeros(6)}
    calls = []
    result = causal_position_outer_bounds(
        linearization, np.zeros(3), {"safe": scope}, on_support=lambda name, event: calls.append((name, event))
    )
    bound = result["scopes"]["safe"]
    check_support_certificate(bound["support_direction"], scope, bound["support"])
    np.testing.assert_allclose(bound["position_lower_bound_m"], 0.0095 - 1e-9, atol=1e-12, rtol=0)
    assert calls == [("safe", "attempted"), ("safe", "completed")]
