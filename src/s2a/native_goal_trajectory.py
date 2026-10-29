"""Causal multi-period robot rollout using native goal reset/latch semantics."""

import numpy as np

from s2a.geometry import finite
from s2a.native_goal_joint_response import rollout_native_goal_joint_response


def rollout_native_goal_trajectory(
    q,
    velocity,
    commands,
    interface,
    initial_joint,
    kp,
    kd,
    dynamics,
    torque_lower,
    torque_upper,
    *,
    latched_rotation_goal,
):
    """Predict 1..16 normalized arm commands from one current robot state.

    Every period advances 25x2ms with state-updated M/J. The next period starts
    from this prediction's final q/v and inherits its last orientation goal;
    it never resets from subsequent observations. The same external dynamics
    callback is used throughout, including its fixed current finger model.

    Commands must satisfy the interface box to its existing 1e-12 roundoff
    tolerance. The interface decoder clips only that admitted roundoff. Rate,
    joint, braking and terminal constraints belong to the caller's audit.
    ``latched_rotation_goal`` is the current finite3x3 historical controller
    goal. Zero rotation retains it; nonzero rotation resets at the predicted
    current SITE. Position resets at each predicted period start in both cases.

    This free robot response does not establish future physical safety or
    task success. No future observations or historical-error updates enter.
    """
    q = finite(q).copy()
    if q.ndim != 1 or len(q) < 6:
        raise ValueError("At least six current robot coordinates required")
    velocity = finite(velocity, q.shape).copy()
    commands = finite(commands).copy()
    if commands.ndim != 2 or commands.shape[1] != 6 or not 1 <= len(commands) <= 16:
        raise ValueError("One to sixteen normalized six-axis commands required")
    lower, upper = finite(interface.input_min, (6,)), finite(interface.input_max, (6,))
    if np.any(lower >= upper) or np.any(commands < lower - 1e-12) or np.any(commands > upper + 1e-12):
        raise ValueError("Commands within the original interface box tolerance required")
    latch = finite(latched_rotation_goal, (3, 3)).copy()
    periods = []
    for command in commands:
        response = rollout_native_goal_joint_response(
            q,
            velocity,
            interface.decode(command),
            initial_joint,
            kp,
            kd,
            dynamics,
            torque_lower,
            torque_upper,
            dt=0.002,
            substeps=25,
            update_matrices=True,
            latched_rotation_goal=latch,
        )
        periods.append(response)
        q = response["joint_position"][-1].copy()
        velocity = response["joint_velocity"][-1].copy()
        latch = response["goal"][:3, :3].copy()
    return dict(
        inner_joint_position=np.concatenate([period["joint_position"] for period in periods]),
        inner_joint_velocity=np.concatenate([period["joint_velocity"] for period in periods]),
        joint_position=np.asarray([period["joint_position"][-1] for period in periods]),
        joint_velocity=np.asarray([period["joint_velocity"][-1] for period in periods]),
        pose=np.asarray([period["endpoint_pose"] for period in periods]),
        twist=np.asarray([period["endpoint_twist"] for period in periods]),
        goals=np.asarray([period["goal"] for period in periods]),
        periods=periods,
    )
