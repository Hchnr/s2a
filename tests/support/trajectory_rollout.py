"""Analytic full-H12 fixtures shared by CPU-only trajectory search tests."""

from types import SimpleNamespace

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import pose_error
from s2a.research.joint_control.limits.joint_preview_constraint_audit import audit_joint_preview_constraints
from s2a.research.trajectory_search.tail_translation_reserve import build_tail_translation_grid


def synthetic_case(*, terminal=None, base=2e-8, prior_max_abs_residual=1e-6, prefix_error=0.0, full_witness=False):
    """Construct a complete saved record with a causal analytic tail response."""
    anchor, previous = np.zeros((12, 6)), np.zeros(6)
    interface = OSCInterface(-np.ones(6), np.ones(6), -np.ones(6), np.ones(6))
    lower, upper, profile = -2 * np.ones(7), 2 * np.ones(7), np.zeros((25, 7))
    desired, tool, normals = np.tile(np.eye(4), (12, 1, 1)), np.eye(4), np.eye(7)[6:]
    scalar = terminal if terminal is not None else lambda commands: base + 0.1 * commands[11, 0]

    def response(commands):
        endpoint_q, endpoint_velocity = np.zeros((12, 7)), np.zeros((12, 7))
        endpoint_velocity[:, 6] = -base
        endpoint_velocity[-1, 6] = -scalar(commands)
        inner_q = np.repeat(endpoint_q, 25, axis=0)
        inner_velocity = np.repeat(endpoint_velocity, 25, axis=0)
        poses, goals = desired.copy(), desired.copy()
        poses[:9, 0, 3] = prefix_error
        if not full_witness:
            poses[9:, 0, 3] = 0.02
        goals[-1, :3, 3] = commands[11, :3]
        periods = []
        for index in range(12):
            torque = np.zeros((25, 7))
            if index == 11:
                torque[:, :3] = commands[11, :3]
            periods.append(
                dict(
                    joint_position=inner_q[index * 25 : (index + 1) * 25].copy(),
                    joint_velocity=inner_velocity[index * 25 : (index + 1) * 25].copy(),
                    requested_torque=torque.copy(),
                    applied_torque=torque.copy(),
                    clipped=np.zeros((25, 7), dtype=bool),
                    mass_change=np.zeros(25),
                    jacobian_change=np.zeros(25),
                    goal=goals[index].copy(),
                    endpoint_pose=poses[index].copy(),
                    endpoint_twist=np.zeros(6),
                    update_matrices=True,
                    dynamics_queries=26,
                )
            )
        return dict(
            inner_joint_position=inner_q,
            inner_joint_velocity=inner_velocity,
            joint_position=endpoint_q,
            joint_velocity=endpoint_velocity,
            pose=poses,
            twist=np.zeros((12, 6)),
            goals=goals,
            periods=periods,
        )

    rollout = response(anchor)
    audit = audit_joint_preview_constraints(
        rollout, anchor, previous, interface, lower, upper, past_inner_error_rad=profile, terminal_normals=normals
    )
    errors = pose_error(desired, rollout["pose"] @ tool)
    squared = np.column_stack(
        (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
    )
    failed = np.flatnonzero(np.any(squared > 1, axis=1))
    prefix = int(failed[0]) if len(failed) else 12
    historical = dict(
        commands=anchor.copy(),
        rollout=rollout,
        audit=audit,
        tcp_errors=errors,
        normalized_squared_errors=squared,
        safety_slack=np.concatenate(
            [
                audit["groups"][key]["slack"].ravel()
                for key in ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")
            ]
        ),
        prefix_length=prefix,
        maximum_normalized_squared_error=float(np.max(squared)),
        maximum_pose_cap_violation=float(max(0, np.max(np.sqrt(squared)) - 1)),
        pose_loss=float(np.sum(squared)),
        feasible=bool(audit["feasible"] and prefix == 12),
    )
    return dict(
        predict=response,
        previous=previous,
        anchor=anchor,
        interface=interface,
        lower=lower,
        upper=upper,
        desired=desired,
        site_from_tcp=tool,
        past_inner_error_rad=profile,
        terminal_normals=normals,
        grid=build_tail_translation_grid(anchor, previous, interface.input_min, interface.input_max),
        historical_anchor=historical,
        prior_max_abs_residual=prior_max_abs_residual,
    )


def source_fixture():
    source = dict(
        current_q=np.arange(7, dtype=float) / 100,
        current_velocity=np.arange(7, dtype=float) / -1000,
        current_quantities=dict(mass=np.eye(7), jacobian=np.zeros((6, 7)), pose=np.eye(4)),
        latched_rotation_goal=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        previous_command=np.zeros(7),
        initial_joint=np.zeros(7),
        kp=np.full(6, 150.0),
        kd=np.full(6, 24.0),
        torque_limits=np.tile([-80.0, 80.0], (7, 1)),
        joint_limits=np.tile([-2.0, 2.0], (7, 1)),
        past_inner_error_rad=np.full((25, 7), 0.001),
        terminal_normals=np.eye(7)[6:],
        desired_tcp=np.repeat(np.eye(4)[None], 12, axis=0),
        site_from_tcp=np.eye(4),
    )
    warm = np.full((12, 6), 0.1)
    warm[0, 3] = -0.0
    interface = SimpleNamespace(input_min=np.full(6, -1.0), input_max=np.full(6, 1.0))
    return source, warm, interface


class CountingDynamics:
    def __init__(self, *, current_query_increment=1):
        self.queries = 0
        self.current_query_increment = current_query_increment
        self.current_inputs = []

    def __call__(self, q, velocity):
        self.queries += self.current_query_increment
        self.current_inputs.append((q.copy(), velocity.copy()))
        q[:] = 100
        velocity[:] = -100
        return dict(mass=np.eye(7), jacobian=np.zeros((6, 7)), pose=np.eye(4))


def captured_rollout(source, calls, *, query_error=0):
    def rollout(
        q, velocity, commands, interface, initial_joint, kp, kd, dynamics, lower, upper, *, latched_rotation_goal
    ):
        calls.append(
            dict(q=q.copy(), velocity=velocity.copy(), commands=commands.copy(), latch=latched_rotation_goal.copy())
        )
        for value, expected in ((initial_joint, source["initial_joint"]), (kp, source["kp"]), (kd, source["kd"])):
            np.testing.assert_array_equal(value, expected)
        np.testing.assert_array_equal(lower, source["torque_limits"][:, 0])
        np.testing.assert_array_equal(upper, source["torque_limits"][:, 1])
        dynamics.queries += 312 + query_error
        q[:] = 9.0
        velocity[:] = -9.0
        latched_rotation_goal[:] = 0.0
        commands[:] = 0.75
        return dict(synthetic_response_index=len(calls))

    return rollout


def complete_case(*, restores=False, base=2e-8):
    case = synthetic_case(base=base) if restores else synthetic_case(base=base, terminal=lambda commands: base)
    source, _, _ = source_fixture()
    source.update(
        previous_command=np.r_[case["previous"], 0.0],
        joint_limits=np.column_stack((case["lower"], case["upper"])),
        desired_tcp=case["desired"].copy(),
        site_from_tcp=case["site_from_tcp"].copy(),
        past_inner_error_rad=case["past_inner_error_rad"].copy(),
        terminal_normals=case["terminal_normals"].copy(),
    )
    return source, case


def complete_rollout(source, case, calls):
    capture = captured_rollout(source, calls)

    def rollout(*args, **kwargs):
        commands = args[2].copy()
        capture(*args, **kwargs)  # Counts312 and deliberately mutates all passed state copies.
        return case["predict"](commands)

    return rollout
