"""Causal preview/refinement transactions over actually completed control periods."""

import time

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import finite
from s2a.research.joint_control.preview.dependency_safe_preview_runtime import solve_preview
from s2a.research.joint_control.tracking.current_coupled_state import rotation_goal_from_command
from s2a.research.native_runtime.development_continuation import current_joint_site_state, current_robot_dynamics
from s2a.research.response_models.forward.current_coupled_response import build_response


class RefinedPreviewFeedback:
    """Prepare and commit exactly the action later observed by both histories."""

    def __init__(
        self,
        env,
        coupled,
        tracking,
        targets_site,
        anchor_site,
        site_from_tcp,
        cursor,
        previous,
        contract,
        receipt,
        *,
        preview_solver=solve_preview,
        dynamics_reader=current_robot_dynamics,
        joint_reader=current_joint_site_state,
    ):
        self.env, self.coupled, self.tracking = env, coupled, tracking
        self.targets = finite(targets_site).copy()
        self.anchor = finite(anchor_site, (4, 4)).copy()
        self.site_from_tcp = finite(site_from_tcp, (4, 4)).copy()
        self.previous = finite(previous, (7,)).copy()
        if self.targets.ndim != 3 or self.targets.shape[1:] != (4, 4) or type(cursor) is not int:
            raise ValueError("One generated chunk and an integer cursor required")
        if not 0 <= cursor <= len(self.targets) - 8:
            raise ValueError("Eight generated targets required at the current cursor")
        if coupled.pending is not None or tracking.phase != "ready":
            raise ValueError("Both histories must be ready at a completed endpoint")
        if coupled.completed_periods != tracking.history.completed_periods:
            raise ValueError("Both histories must cover the same completed periods")
        self.cursor, self.contract, self.receipt = cursor, contract, receipt
        self.initial_periods = coupled.completed_periods
        self.interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        self.preview_solver, self.dynamics_reader, self.joint_reader = preview_solver, dynamics_reader, joint_reader
        self.pending = None
        self.receipt["preview_calls"] = {"attempted": 0, "completed": 0}
        self.receipt["refiner_calls"] = {"attempted": 0, "completed": 0}
        self.receipt["refiner_response_evaluations"] = 0

    def select(self, q, limits, jacobian, previous, offset):
        if self.pending is not None or self.tracking.phase != "ready" or self.coupled.pending is not None:
            raise ValueError("Previous action must be observed before another selection")
        if self.coupled.completed_periods != self.tracking.history.completed_periods:
            raise ValueError("Both histories must cover the same completed periods")
        if type(offset) is not int or offset != self.coupled.completed_periods - self.initial_periods:
            raise ValueError("Selection must follow completed control periods")
        previous = finite(previous, (7,))
        if not np.array_equal(previous, self.previous):
            raise ValueError("Previous command must equal the actual last dispatch")
        q, limits, jacobian = finite(q, (7,)), finite(limits, (7, 2)), finite(jacobian, (6, 7))
        mass, qvel, site = self.dynamics_reader(self.env, jacobian)
        twist = jacobian @ qvel
        controller = self.env.robots[0].controller
        innovation = self.coupled.current(site, twist)
        started = time.monotonic()
        self.receipt["preview_calls"]["attempted"] += 1
        seed = self.preview_solver(
            mass,
            jacobian,
            site,
            twist,
            self.targets,
            self.anchor,
            self.cursor + offset,
            previous[:6],
            self.contract,
            innovation=innovation,
        )
        self.receipt["preview_calls"]["completed"] += 1
        dynamics = {
            "mass": mass,
            "jacobian": jacobian,
            "robot_joint_position": q,
            "robot_joint_velocity": qvel,
            "kp": np.asarray(controller.kp).copy(),
            "kd": np.asarray(controller.kd).copy(),
        }
        self.receipt["refiner_calls"]["attempted"] += 1
        refined = self.tracking.prepare(
            self.env,
            dynamics,
            site,
            twist,
            self.targets,
            self.anchor,
            self.cursor + offset,
            seed["preview"]["planned_commands"],
            previous[:6],
            self.contract,
            site_from_tcp=self.site_from_tcp,
        )
        self.receipt["refiner_calls"]["completed"] += 1
        evaluations = int(refined["evaluations"])
        if not 0 <= evaluations <= 480:
            raise ValueError("Original per-period refiner response budget exceeded")
        self.receipt["refiner_response_evaluations"] += evaluations
        command = finite(refined["command"], (6,)).copy()
        goal, relative = rotation_goal_from_command(command, self.interface, site[:3, :3], controller.goal_ori)
        np.testing.assert_allclose(goal, refined["expected_goal_ori"], atol=1e-8, rtol=0)
        model = build_response(mass, jacobian, controller.kp, controller.kd, relative)
        self.tracking.commit(command)
        self.coupled.commit_executed(site, twist, model, command[:3], self.contract)
        dispatched = np.r_[command, 0.0]
        self.pending = {
            "command": dispatched.copy(),
            "goal_pos": np.asarray(refined["expected_goal_pos"]).copy(),
            "goal_ori": np.asarray(refined["expected_goal_ori"]).copy(),
            "offset": offset,
        }
        return {
            "command": dispatched,
            "newly_selected": True,
            "preview_command": seed["command"],
            "refined_plan": refined["planned_commands"],
            "route": refined["route"],
            "refiner_evaluations": evaluations,
            "optimization": refined["optimization"],
            "decision_seconds": time.monotonic() - started,
            "predecision_minimum_margin_rad": float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))),
        }

    def observe(self, env, command, offset):
        if env is not self.env or self.pending is None or offset != self.pending["offset"]:
            raise ValueError("One committed action at the expected offset must be observed")
        command = finite(command, (7,))
        if not np.array_equal(command, self.pending["command"]):
            raise ValueError("Observed command differs from the committed action")
        controller = env.robots[0].controller
        np.testing.assert_allclose(controller.goal_pos, self.pending["goal_pos"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(controller.goal_ori, self.pending["goal_ori"], atol=1e-8, rtol=0)
        _q, _limits, jacobian = self.joint_reader(env)
        _mass, qvel, site = self.dynamics_reader(env, jacobian)
        twist = jacobian @ qvel
        coupled = self.coupled.observe_completed(site, twist)
        tracking = self.tracking.observe(site, twist)
        self.previous = command.copy()
        self.pending = None
        return {
            "coupled_innovation": coupled,
            "tracking_innovation": tracking,
            "completed_periods": self.coupled.completed_periods,
            "completed_history_ready": self.tracking.phase == "ready" and self.coupled.pending is None,
        }
