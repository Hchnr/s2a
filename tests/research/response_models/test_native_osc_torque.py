"""Physical algebra checks, independent of optional simulator installations."""

import numpy as np
from numpy.testing import assert_array_equal

from s2a.research.response_models.forward.native_osc_torque import replay_order_uncoupled_torque


def inputs():
    return dict(
        mass=np.diag(np.arange(1.0, 8.0)),
        jacobian=np.eye(6, 7),
        pose=np.eye(4),
        goal=np.eye(4),
        q=np.zeros(7),
        velocity=np.zeros(7),
        initial_joint=np.zeros(7),
        kp=np.ones(6),
        kd=np.ones(6),
    )


def test_decoupled_translation_and_sine_rotation_use_site_axes():
    state = inputs()
    state["goal"][:3, 3] = [0.25, -0.5, 0.125]
    state["goal"][:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # Diagonal operational inertia equals the corresponding joint mass.
    assert_array_equal(replay_order_uncoupled_torque(**state), [0.25, -1, 0.375, 0, 0, 6, 0])


def test_redundant_joint_spring_and_damping_do_not_enter_task_axes():
    state = inputs()
    state["initial_joint"][6] = 0.5
    state["velocity"][6] = 0.25
    assert_array_equal(replay_order_uncoupled_torque(**state, null_kp=4), [0, 0, 0, 0, 0, 0, 7])
    assert_array_equal(replay_order_uncoupled_torque(**state, null_kp=0), np.zeros(7))


def test_rank_zero_task_leaves_mass_weighted_nullspace_spring_and_preserves_inputs():
    state = inputs()
    state["jacobian"][:] = 0
    state["initial_joint"][:] = 0.5
    state["velocity"][:] = 0.25
    before = {key: value.copy() for key, value in state.items()}
    assert_array_equal(replay_order_uncoupled_torque(**state, null_kp=4), np.arange(1.0, 8.0))
    for key in state:
        assert_array_equal(state[key], before[key])
