import numpy as np
from executed_residual_independent import ExecutedResidualReplay

from s2a.executed_command_residual import ExecutedCommandResidualOSC
from s2a.osc import OSCInterface


def test_independent_scalar_history_matches_matrix_response_after_actual_blending():
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6) * 0.05, np.ones(6) * 0.05)
    kp, kd = np.arange(6) * 10 + 120.0, np.arange(6) + 22.0
    producer = ExecutedCommandResidualOSC(interface, kp, kd, 0.05, max_command_rate=5)
    verifier = ExecutedResidualReplay(interface, kp, kd, 0.05, max_command_rate=5)
    rng = np.random.default_rng(42)
    pose, target = np.eye(4), np.eye(4)
    velocity = np.zeros(6)
    actual = np.zeros(6)
    for _ in range(25):
        target[:3, 3] = pose[:3, 3] + rng.normal(size=3) * 0.01
        reference = rng.normal(size=3) * 0.02
        a = producer.convert(pose, target, velocity, desired_linear_velocity=reference)
        b = verifier.convert(pose, target, velocity, desired_linear_velocity=reference)
        np.testing.assert_allclose(a["command"], b["command"], atol=1e-12, rtol=0)
        for key in ("position", "linear_velocity"):
            np.testing.assert_allclose(a["response_residual"][key], b["response_residual"][key], atol=1e-12, rtol=0)
        actual = np.clip(
            0.5 * a["command"] + 0.5 * rng.uniform(-1, 1, 6),
            np.maximum(-1, actual - 0.25),
            np.minimum(1, actual + 0.25),
        )
        pa = producer.commit_executed(pose, velocity, actual)
        pb = verifier.commit_executed(pose, velocity, actual)
        for key in pa:
            np.testing.assert_allclose(pa[key], pb[key], atol=1e-12, rtol=0)
        pose[:3, 3] += pa["nominal_motion"] + rng.normal(size=3) * 0.001
        velocity = np.r_[pa["nominal_linear_velocity"] + rng.normal(size=3) * 0.01, np.zeros(3)]
