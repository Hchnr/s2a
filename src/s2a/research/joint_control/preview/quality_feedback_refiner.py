"""Apply first-period quality constraints to the causal native feedback plan."""

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.geometry import finite, pose_error
from s2a.references.generated_twist import reference_world_twist
from s2a.research.joint_control.native.tracking_margin_native import refine_native_preview
from s2a.research.joint_control.preview.interior_joint_preview import build_joint_preview, rollout_joint_preview
from s2a.research.joint_control.tracking.current_coupled_state import rotation_goal_from_command
from s2a.research.joint_control.tracking.early_quality_diagnostic import independently_check_witness
from s2a.research.response_models.causal.causal_inner_error import CausalInnerError
from s2a.research.trajectory_search.quality_capped_inner_joint_plan import solve_interior_plan


class QualitySelectionStopped(RuntimeError):
    """A model plan could not be authorized for dispatch."""

    def __init__(self, reason, status, detail=""):
        super().__init__(detail or status)
        self.reason, self.status, self.detail = reason, status, detail

    def decision(self):
        return {"stop": {"reason": self.reason, "solver_status": self.status, "detail": self.detail}}


class QualityFeedbackRefiner:
    """Replace the staged baseline only after a past-only witness passes audit."""

    def __init__(
        self,
        history,
        receipt,
        *,
        start_step,
        start_cursor,
        baseline=refine_native_preview,
        solver=solve_interior_plan,
        auditor=independently_check_witness,
    ):
        if not isinstance(history, CausalInnerError) or type(start_step) is not int or type(start_cursor) is not int:
            raise ValueError("Causal inner-error history and fixed start indices required")
        if start_step < 0 or start_cursor < 0:
            raise ValueError("Nonnegative start indices required")
        self.history, self.receipt = history, receipt
        self.start_step, self.start_cursor = start_step, start_cursor
        self.baseline, self.solver, self.auditor = baseline, solver, auditor
        self.stats = receipt["quality_refiner"] = {
            "baseline_calls_attempted": 0,
            "baseline_calls_completed": 0,
            "baseline_response_evaluations": 0,
            "quality_calls_attempted": 0,
            "quality_calls_completed": 0,
            "quality_response_evaluations": 0,
            "witness_audits_attempted": 0,
            "witness_audits_passed": 0,
            "endpoint_queries_charged": 0,
            "transition_steps_charged": 0,
        }

    def __call__(
        self,
        env,
        dynamics,
        site,
        twist,
        poses,
        anchor,
        index,
        seed_plan,
        previous,
        contract,
        plan_history,
        *,
        site_from_tcp,
    ):
        offset = index - self.start_cursor
        past = self.history.snapshot()
        if offset < 0 or past["completed_periods"] != offset:
            raise ValueError("Only this branch's completed inner periods may tighten the current decision")
        self.stats["baseline_calls_attempted"] += 1
        baseline = self.baseline(
            env,
            dynamics,
            site,
            twist,
            poses,
            anchor,
            index,
            seed_plan,
            previous,
            contract,
            plan_history,
            site_from_tcp=site_from_tcp,
        )
        self.stats["baseline_calls_completed"] += 1
        evaluations = int(baseline["evaluations"])
        if not 0 <= evaluations <= 480:
            raise ValueError("Original per-period refiner response budget exceeded")
        self.stats["baseline_response_evaluations"] += evaluations
        seed = finite(baseline["planned_commands"])
        if plan_history.pending_plan is None or not np.array_equal(plan_history.pending_plan, seed):
            raise ValueError("Baseline must stage exactly its proposed plan")

        controller = env.robots[0].controller
        limits = np.asarray(
            [env.sim.model.jnt_range[env.sim.model.joint_name2id(f"robot0_joint{k}")] for k in range(1, 8)]
        )
        mass, jacobian, q, velocity = (
            dynamics[k] for k in ("mass", "jacobian", "robot_joint_position", "robot_joint_velocity")
        )
        kp, kd = dynamics["kp"], dynamics["kd"]
        model = build_joint_preview(mass, jacobian, kp, kd, q, controller.initial_joint, null_kp=10.0, dt=0.002)
        interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
        tcp_poses, tcp_anchor = poses @ site_from_tcp, anchor @ site_from_tcp
        desired = tcp_poses[index : index + len(seed)]
        if len(desired) != len(seed):
            raise ValueError("Generated references must cover the entire selected plan")
        twists = np.asarray(
            [reference_world_twist(tcp_poses, j, tcp_anchor, 0.05)["twist"] for j in range(index, index + len(seed))]
        )
        self.stats["quality_calls_attempted"] += 1
        solved = self.solver(
            model,
            velocity,
            seed,
            previous,
            interface,
            site,
            site_from_tcp,
            desired,
            twists,
            limits[:, 0],
            limits[:, 1],
            past_inner_error_rad=past["inner_position_error_rad"],
        )
        self.stats["quality_calls_completed"] += 1
        count = int(solved["response_evaluations"])
        if not 0 <= count <= 160:
            raise ValueError("Per-period quality response budget exceeded")
        self.stats["quality_response_evaluations"] += count
        if solved["witness_found"] is not True:
            plan_history.pending_plan = None
            raise QualitySelectionStopped("quality_no_witness", str(solved["status"]))

        self.stats["witness_audits_attempted"] += 1
        # Charge the full audit even if a check fails partway through.
        self.stats["endpoint_queries_charged"] += 25 * len(seed)
        self.stats["transition_steps_charged"] += 325 * len(seed)
        try:
            verification = self.auditor(
                solved,
                mass,
                jacobian,
                kp,
                kd,
                q,
                velocity,
                controller.initial_joint,
                interface,
                previous,
                site,
                site_from_tcp,
                desired[0],
                limits[:, 0],
                limits[:, 1],
                past_inner_error_rad=past["inner_position_error_rad"],
            )
            if verification.get("passed") is not True:
                raise ValueError("Independent witness audit did not pass")
        except (AssertionError, ValueError) as error:
            plan_history.pending_plan = None
            raise QualitySelectionStopped("quality_witness_rejected", str(solved["status"]), str(error)) from error
        self.stats["witness_audits_passed"] += 1
        plan = finite(solved["planned_commands"], seed.shape).copy()
        nominal = rollout_joint_preview(
            model,
            velocity,
            np.asarray([interface.decode(plan[0])]),
            site,
            limits[:, 0],
            limits[:, 1],
            enforce_limits=True,
        )
        np.testing.assert_allclose(nominal["pose"][0], solved["rollout"]["pose"][0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(nominal["twist"][0], solved["rollout"]["twist"][0], atol=1e-10, rtol=0)
        goal, _ = rotation_goal_from_command(plan[0], interface, site[:3, :3], controller.goal_ori)
        plan_history.pending_plan = None
        plan_history.stage_plan(plan)
        self.history.stage(self.start_step + offset, solved["rollout"]["inner_joint_position"][:25])
        result = dict(baseline)
        result.update(
            command=plan[0].copy(),
            planned_commands=plan,
            nominal_response=np.r_[pose_error(nominal["pose"][0], site), nominal["twist"][0]],
            route="quality_witness",
            expected_goal_pos=site[:3, 3] + interface.decode(plan[0])[:3],
            expected_goal_ori=goal,
            quality={
                "status": solved["status"],
                "response_evaluations": count,
                "first_tcp_error": solved["first_tcp_error"],
                "past_error": past,
                "verification": verification,
            },
        )
        return result
