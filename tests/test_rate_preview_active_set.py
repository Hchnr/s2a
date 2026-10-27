import numpy as np
import pytest
from rate_preview_independent import verify_plan
from test_direct_arm_inverse import parameters

from s2a.rate_constrained_preview_active_set import plan_translation


def contract():
    p = parameters()
    return dict(
        control_freq=20,
        control_frame="MuJoCo world",
        kp=np.full(6, 150.0),
        kd=np.full(6, 25.0),
        **{
            k: np.r_[p["translation_" + k], p["translation_" + k]]
            for k in ("input_min", "input_max", "output_min", "output_max")
        },
    )


def test_one_period_matches_unconstrained_position_velocity_least_squares():
    from pose_velocity_arm_independent_replay import PoseVelocityArmReplay

    from s2a.osc import OSCInterface

    c = contract()
    interface = OSCInterface(*(c[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    inverse = PoseVelocityArmReplay(interface, c["kp"], c["kd"], 0.05)
    position, velocity = np.zeros(3), np.full(3, 0.001)
    references, targets = np.full((1, 3), 0.0003), np.full((1, 3), 0.002)
    plan = plan_translation(position, velocity, references, targets, np.zeros(3), c, innovation=np.zeros((2, 3)))
    current, desired = np.eye(4), np.eye(4)
    desired[:3, 3] = references[0]
    expected = inverse.convert(current, desired, np.r_[velocity, np.zeros(3)], desired_linear_velocity=targets[0])
    np.testing.assert_allclose(plan["command"], expected["command"][:3], atol=1e-10, rtol=0)
    verify_plan(plan, position, velocity, references, targets, np.zeros(3), c, np.zeros((2, 3)))


def test_generated_preview_changes_current_command_and_keeps_all_rate_constraints():
    c = contract()
    a = np.full((6, 3), 0.004)
    b = a.copy()
    b[1:] = -0.004
    kwargs = dict(
        position=np.zeros(3),
        velocity=np.zeros(3),
        reference_velocities=np.zeros((6, 3)),
        previous_command=np.zeros(3),
        contract=c,
        innovation=np.zeros((2, 3)),
    )
    first = plan_translation(references=a, **kwargs)
    second = plan_translation(references=b, **kwargs)
    assert np.linalg.norm(first["command"] - second["command"]) > 0.01
    for plan, refs in ((first, a), (second, b)):
        assert np.max(np.abs(np.diff(np.vstack([np.zeros(3), plan["planned_commands"]]), axis=0))) <= 0.25 + 1e-10
        verify_plan(plan, np.zeros(3), np.zeros(3), refs, np.zeros((6, 3)), np.zeros(3), c, np.zeros((2, 3)))


def test_random_bounded_innovations_and_shortened_horizons_have_independent_kkt():
    rng = np.random.default_rng(42)
    c = contract()
    for horizon in range(1, 7):
        position = rng.uniform(-0.2, 0.2, 3)
        velocity = rng.normal(0, 0.05, 3)
        refs = position + rng.normal(0, 0.01, (horizon, 3))
        targets = rng.normal(0, 0.05, (horizon, 3))
        previous = rng.uniform(-1, 1, 3)
        innovation = rng.normal(0, [0.0001, 0.005], (3, 2)).T
        plan = plan_translation(position, velocity, refs, targets, previous, c, innovation=innovation)
        verify_plan(plan, position, velocity, refs, targets, previous, c, innovation)


def test_unavailable_preview_or_infeasible_actual_previous_command_is_rejected():
    c = contract()
    with pytest.raises(ValueError, match="references"):
        plan_translation(
            np.zeros(3), np.zeros(3), np.zeros((7, 3)), np.zeros((7, 3)), np.zeros(3), c, innovation=np.zeros((2, 3))
        )
    with pytest.raises(ValueError, match="previous"):
        plan_translation(
            np.zeros(3),
            np.zeros(3),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.full(3, 2.0),
            c,
            innovation=np.zeros((2, 3)),
        )
