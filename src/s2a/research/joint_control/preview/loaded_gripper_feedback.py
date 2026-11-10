"""Bind causal arm feedback and gripper state to one completed physical action."""

import numpy as np

from s2a.geometry import inverse, validate_transform
from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner, QualitySelectionStopped
from s2a.research.native_runtime.development_continuation import current_robot_dynamics


def current_gripper_state(env):
    """Read the refreshed measured aperture, velocity, and actual motor target."""
    robot, data = env.robots[0], env.sim.data
    q = np.asarray(data.qpos)[robot._ref_gripper_joint_pos_indexes].copy()
    velocity = np.asarray(data.qvel)[robot._ref_gripper_joint_vel_indexes].copy()
    motor = np.asarray(robot.gripper.current_action, dtype=float).copy()
    if (
        q.shape != (2,)
        or velocity.shape != (2,)
        or motor.shape != (2,)
        or not np.isfinite([*q, *velocity, *motor]).all()
    ):
        raise ValueError("Finite measured Panda gripper state and motor target required")
    return {
        "opening": float(np.clip((q[0] - q[1] - 0.001) / 0.079, 0, 1)),
        "period_velocity": float(0.05 * (velocity[0] - velocity[1]) / 0.079),
        "motor": motor,
        "qpos": q,
        "qvel": velocity,
    }


class LoadedGripperFeedback:
    """Use one generated reference and commit gripper state after dispatch."""

    def __init__(
        self,
        env,
        arm_policy,
        gripper,
        inner_history,
        poses,
        openings,
        cursor,
        site_from_tcp,
        start_step,
        receipt,
        *,
        dynamics_reader=current_robot_dynamics,
        gripper_reader=current_gripper_state,
        initial_inner_loss=False,
        max_steps=8,
    ):
        self.env, self.arm_policy, self.gripper, self.inner_history = env, arm_policy, gripper, inner_history
        self.poses, self.openings = np.asarray(poses, dtype=float).copy(), np.asarray(openings, dtype=float).copy()
        self.site_from_tcp = np.asarray(site_from_tcp, dtype=float).copy()
        if (
            self.poses.shape != (40, 4, 4)
            or self.openings.shape != (40,)
            or self.site_from_tcp.shape != (4, 4)
            or not np.isfinite(self.poses).all()
            or not np.isfinite(self.openings).all()
            or not np.isfinite(self.site_from_tcp).all()
            or np.any((self.openings < 0) | (self.openings > 1))
            or type(cursor) is not int
            or type(max_steps) is not int
            or max_steps not in (8, 16)
            or not 0 <= cursor
            or cursor + max_steps + 7 > 40
            or type(start_step) is not int
            or start_step < 0
            or type(initial_inner_loss) is not bool
        ):
            raise ValueError("One finite generated TCP/opening chunk and aligned cursor are required")
        self.cursor, self.start_step, self.receipt, self.max_steps = cursor, start_step, receipt, max_steps
        self.dynamics_reader, self.gripper_reader = dynamics_reader, gripper_reader
        self.pending, self.last_inner_loss = None, initial_inner_loss
        self.receipt["gripper_proposals"] = {
            "attempted": 0,
            "completed": 0,
            "planner_calls": 0,
            "candidate_sequences": 0,
            "committed": 0,
            "aborted_before_dispatch": 0,
        }

    def rebind_reference(self, poses, openings, anchor, *, offset, next_cursor, generated_step):
        """Switch only a generated reference at an already completed boundary."""
        arm, past, gripper_state = self.arm_policy, self.inner_history.snapshot(), self.gripper.state()
        if (
            self.pending is not None
            or self.last_inner_loss
            or gripper_state.get("staged") is not False
            or gripper_state.get("faulted") is not False
            or arm.pending is not None
            or arm.coupled.pending is not None
            or arm.tracking.phase != "ready"
            or type(offset) is not int
            or not 0 < offset < self.max_steps
            or arm.coupled.completed_periods != arm.initial_periods + offset
            or arm.tracking.history.completed_periods != arm.coupled.completed_periods
            or past["completed_periods"] != offset
            or past["last_completed_step"] != self.start_step + offset - 1
            or self.inner_history._prediction is not None
        ):
            raise ValueError("A fully observed matching action history is required before switching")
        refiner = arm.tracking._refiner
        if (
            not isinstance(refiner, QualityFeedbackRefiner)
            or refiner.history is not self.inner_history
            or refiner.receipt is not self.receipt
            or refiner.start_step != self.start_step
            or refiner.start_cursor != self.cursor
            or arm.cursor != self.cursor
        ):
            raise ValueError("The original quality and reference history must remain bound")
        if (
            type(next_cursor) is not int
            or next_cursor < offset
            or type(generated_step) is not int
            or generated_step < 0
            or generated_step + next_cursor != self.start_step + offset
        ):
            raise ValueError("The new cursor must preserve absolute endpoint timing")
        base = next_cursor - offset
        if base + self.max_steps + 7 > 40:
            raise ValueError("The new generated reference cannot cover the remaining preview")
        poses = validate_transform(np.asarray(poses, dtype=float))
        anchor = validate_transform(np.asarray(anchor, dtype=float))
        openings = np.asarray(openings, dtype=float)
        if (
            poses.shape != (40, 4, 4)
            or anchor.shape != (4, 4)
            or openings.shape != (40,)
            or not np.isfinite(openings).all()
            or np.any((openings < 0) | (openings > 1))
        ):
            raise ValueError("A complete generated TCP/opening reference is required")
        site_targets = poses @ inverse(self.site_from_tcp)
        site_anchor = anchor @ inverse(self.site_from_tcp)
        self.poses, self.openings, self.cursor = poses.copy(), openings.copy(), base
        arm.targets, arm.anchor, arm.cursor = site_targets.copy(), site_anchor.copy(), base
        refiner.start_cursor = base
        return {
            "completed_offset": offset,
            "next_cursor": next_cursor,
            "generated_step": generated_step,
            "cursor_base": base,
        }

    def select(self, q, limits, jacobian, previous, offset):
        if self.pending is not None:
            raise ValueError("The previous physical action must complete first")
        q, limits = np.asarray(q, dtype=float), np.asarray(limits, dtype=float)
        previous = np.asarray(previous, dtype=float)
        if (
            q.shape != (7,)
            or limits.shape != (7, 2)
            or previous.shape != (7,)
            or not np.isfinite(q).all()
            or not np.isfinite(limits).all()
            or not np.isfinite(previous).all()
            or np.any(limits[:, 0] >= limits[:, 1])
            or type(offset) is not int
            or not 0 <= offset < self.max_steps
        ):
            raise ValueError("Finite joint state, completed command and bounded offset required")
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        if margin < 0.05 or self.last_inner_loss:
            return {"stop": {"reason": "observed_joint_buffer_loss", "minimum_current_margin_rad": margin}}
        _mass, _velocity, site = self.dynamics_reader(self.env, jacobian)
        gripper = self.gripper_reader(self.env)
        stats = self.receipt["gripper_proposals"]
        stats["attempted"] += 1
        token = self.gripper.prepare(
            gripper["opening"],
            gripper["motor"],
            gripper["period_velocity"],
            self.poses,
            self.openings,
            self.cursor + offset,
            np.asarray(site) @ self.site_from_tcp,
            previous_command=float(previous[6]),
            dt=0.05,
        )
        stats["completed"] += 1
        plan = token["detail"]["plan"]
        if plan is not None:
            stats["planner_calls"] += 1
            stats["candidate_sequences"] += (
                plan["valid_sequences"] + plan["rejected_nonfinite_sequences"] + plan["rejected_velocity_sequences"]
            )
        try:
            decision = self.arm_policy.select(q, limits, jacobian, previous, offset, gripper_command=token["command"])
        except QualitySelectionStopped as error:
            self.gripper.abort(token)
            stats["aborted_before_dispatch"] += 1
            stopped = error.decision()
            stopped["stop"]["gripper_proposal"] = token["detail"]
            stopped["stop"]["gripper_proposal_dispatched"] = False
            return stopped
        except BaseException:
            self.gripper.abort(token)
            stats["aborted_before_dispatch"] += 1
            raise
        command = np.asarray(decision["command"], dtype=float)
        if command.shape != (7,) or command[6] != token["command"]:
            self.gripper.abort(token)
            stats["aborted_before_dispatch"] += 1
            raise ValueError("Arm and gripper must stage the same complete command")
        self.pending = {"token": token, "command": command.copy(), "limits": limits.copy(), "offset": offset}
        return {**decision, "gripper_decision": token["detail"]}

    def complete(self, env, command, offset, actual_inner):
        if self.pending is None or env is not self.env or offset != self.pending["offset"]:
            raise ValueError("One matching staged action must physically complete")
        command = np.asarray(command, dtype=float)
        if not np.array_equal(command, self.pending["command"]):
            raise ValueError("The completed physical command differs from the staged command")
        actual = np.asarray(actual_inner, dtype=float)
        if actual.shape != (25, 7) or not np.isfinite(actual).all():
            raise ValueError("All 25 completed inner joint samples are required")
        feedback = self.arm_policy.observe(env, command, offset)
        self.gripper.commit(self.pending["token"], actual_command=float(command[6]), completed_steps=1)
        residual = self.inner_history.complete(self.start_step + offset, actual)
        limits = self.pending["limits"]
        self.last_inner_loss = bool(np.any(np.minimum(actual - limits[:, 0], limits[:, 1] - actual) < 0.05))
        self.pending = None
        self.receipt["gripper_proposals"]["committed"] += 1
        return {
            "arm_history": feedback,
            "gripper_history": self.gripper.state(),
            "completed_inner_error_rad": residual,
            "past_inner_error": self.inner_history.snapshot(),
        }
