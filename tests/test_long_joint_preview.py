"""Long preview preserves short semantics and reanchored joint dynamics."""

import numpy as np

from s2a.joint_limit_preview import rollout_joint_preview as short_rollout
from s2a.joint_limit_response import build_joint_response, predict_joint_response
from s2a.long_joint_preview import build_joint_preview, rollout_joint_preview


def test_long_preview_reanchors_and_carries_velocity_across_eight_periods():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = np.eye(7) + a.T @ a * 0.03
    j = np.c_[np.eye(6), rng.normal(size=6) * 0.1]
    q, initial = rng.normal(size=(2, 7)) * 0.1
    velocity = rng.normal(size=7) * 0.02
    deltas = rng.normal(size=(8, 6)) * 0.02
    kp, kd = np.ones(6) * 100, np.ones(6) * 20
    model = build_joint_preview(mass, j, kp, kd, q, initial, null_kp=10.0, dt=0.002)
    lo, hi = -np.ones(7) * 2, np.ones(7) * 2
    actual = rollout_joint_preview(model, velocity, deltas, np.eye(4), lo, hi, enforce_limits=False)
    short = short_rollout(model, velocity, deltas[:4], np.eye(4), lo, hi, enforce_limits=False)
    for key in ("pose", "joint_position", "joint_velocity"):
        np.testing.assert_array_equal(actual[key][:4], short[key])
    for k, delta in enumerate(deltas):
        rebuilt = build_joint_response(mass, j, kp, kd, q, initial, null_kp=10.0, dt=0.002)
        expected = predict_joint_response(rebuilt, velocity, delta, lo, hi, substeps=25, enforce_limits=False)
        np.testing.assert_allclose(actual["joint_position"][k], expected["joint_position"], atol=1e-11, rtol=0)
        q, velocity = expected["joint_position"], expected["joint_velocity"]
    # A late input cannot affect earlier endpoints; its final response derivative
    # must match an independent centered perturbation of the physical command.
    np.testing.assert_array_equal(actual["joint_position_jacobian"][:7, :, 42:], np.zeros((7, 7, 6)))
    plus, minus = deltas.copy(), deltas.copy()
    plus[7, 0] += 1e-6
    minus[7, 0] -= 1e-6
    p = rollout_joint_preview(model, rng.normal(size=7) * 0, plus, np.eye(4), lo, hi, enforce_limits=False)
    m = rollout_joint_preview(model, np.zeros(7), minus, np.eye(4), lo, hi, enforce_limits=False)
    # Free-model derivatives do not depend on initial velocity.
    np.testing.assert_allclose(
        (p["joint_position"][-1] - m["joint_position"][-1]) / 2e-6,
        actual["joint_position_jacobian"][-1, :, 42],
        atol=1e-8,
        rtol=1e-5,
    )


def test_eight_period_inverse_recovers_feasible_action_generated_reference():
    from s2a.long_joint_inverse import refine_long_plan
    from s2a.osc import OSCInterface

    q = np.zeros(7)
    model = build_joint_preview(
        np.eye(7), np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100, np.ones(6) * 20, q, q, null_kp=10.0, dt=0.002
    )
    known = np.zeros((8, 6))
    known[:, 0] = [0.02, 0.04, 0.06, 0.08, 0.08, 0.06, 0.04, 0.02]
    known[:, 3] = 1e-9
    seed = np.zeros_like(known)
    seed[:, 3] = 1e-9
    lo, hi = -np.ones(7), np.ones(7)
    target = rollout_joint_preview(model, np.zeros(7), known, np.eye(4), lo, hi, enforce_limits=True)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    result = refine_long_plan(
        model, np.zeros(7), seed, np.zeros(6), interface, np.eye(4), np.eye(4), target["pose"], target["twist"], lo, hi
    )
    assert result["objective"] < result["seed_objective"] * 1e-6
    np.testing.assert_allclose(result["planned_commands"][:, 0], known[:, 0], atol=1e-3, rtol=0)
    assert np.max(abs(np.diff(np.vstack((np.zeros(6), result["planned_commands"])), axis=0))) <= 0.25 + 1e-12
