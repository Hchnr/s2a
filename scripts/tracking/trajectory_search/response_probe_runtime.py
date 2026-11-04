"""Shared finite CPU response/independent-replay protocol for fixed probe suites.

No optimizer, acceptance rule or candidate selection lives here. Callers must
freeze commands and budgets before constructing models; unsafe numerical cases
remain observations and never become dispatched robot actions.
"""

import copy
import os
import traceback

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.dynamics.mujoco_robot import MujocoRobotDynamics
from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result, result_json
from s2a.research.trajectory_search.accepted_point_sqp_evidence import PREFIX_CHECK_REF
from s2a.research.trajectory_search.constraint_residual_repair import evaluate_repair_entry
from s2a.research.trajectory_search.inputs import model_registration
from s2a.research.trajectory_search.replay_order_trajectory import rollout_replay_order_trajectory
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import (
    CPU_ENVIRONMENT,
    REPO,
    checked,
    read,
    reference,
    remember,
    require,
)
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values
from scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery import load_inputs
from scripts.tracking.trajectory_search.probe_tracking_v5_terminal_only_tail_repair import load_environment_guard
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import check_full_candidate
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare


class ResponseProbeRuntime:
    """Current measured state/history and a separately owned finite case list."""

    def __init__(self, output, qa_path):
        strict_native_environment()
        require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Single-thread CPU suite")
        require(not output.exists(), "Exclusive output; no overwrite or automatic retry")
        qa_ref = reference(qa_path)
        qa = read(qa_ref)
        require(
            qa["passed"] and qa["all_tests_passed"] and qa["source_unchanged"] and qa["skipped_count"] == 0,
            "Complete passing corresponding QA",
        )
        self.before = snapshot(REPO)
        qa_sources = read(qa["source_after_ref"])
        for key in ["sources", "current_map_ref", "driver_ref"]:
            require(self.before[key] == qa_sources[key], "Current source matches QA: " + key)
        self.output = output
        self.bundle, _, self.context = load_inputs()
        self.refs = self.context.runtime_refs
        self.qa_ref = remember(qa_ref, self.refs)
        remember(qa["source_after_ref"], self.refs)
        self.source = copy.deepcopy(self.bundle["source"])
        self.source_reg = self.bundle["source_registration"]
        self.control = self.bundle["control"]
        self.model_reg, self.proof = model_registration(self.source_reg, self.context, self.bundle)
        control = read(remember(self.source_reg["control_ref"], self.refs))
        tool = read(remember(self.source_reg["tool_ref"], self.refs))
        self.source["site_from_tcp"] = np.linalg.solve(
            np.asarray(control["source_from_site"]), np.asarray(tool["source_from_tcp"])
        )
        self.guard = load_environment_guard(self.refs)
        prefix = read(remember(PREFIX_CHECK_REF, self.refs))
        self.prefix_reg = read(remember(prefix["registration_ref"], self.refs))

    def load(self, ref, sha=None):
        if sha is not None:
            require(ref["sha256"] == sha, "Exact preregistered input")
        return read(remember(ref, self.refs))

    def run(self, cases, expected_anchor, config, plan, source_paths, detail):
        """Evaluate every declared H12 once per side; compare the full audit."""
        require(len(cases) >= 2 and len(cases) == detail["response_count_per_side"], "Exact frozen case count")
        require(cases[0]["role"] == "anchor" and cases[-1]["role"] == "anchor_repeat", "Mandatory repeat roles")
        require(all(np.asarray(case["commands"]).shape == (12, 6) for case in cases), "Complete H12 commands only")
        require(
            np.asarray(cases[0]["commands"]).tobytes() == np.asarray(cases[-1]["commands"]).tobytes(),
            "Repeat uses the exact initial command",
        )
        require(
            detail["total_query_budget"] == 2 + 624 * len(cases)
            and detail["total_inner_step_budget"] == 600 * len(cases),
            "Budgets cover both complete response suites and initial queries",
        )
        self.context.finish()
        registration = publish_result(
            self.output / "registration.json",
            dict(
                experiment=detail["experiment"],
                config_ref=reference(config),
                plan_ref=reference(plan),
                config=read(reference(config)),
                cases=cases,
                detail=detail,
                qa_ref=self.qa_ref,
                source_refs=[reference(path) for path in source_paths],
                input_refs=list(self.refs.values()),
                source_snapshot_ref=publish_result(self.output / "source_before.json", self.before),
                asset_proof=self.proof,
            ),
        )
        result = dict(
            complete=False,
            passed=False,
            registration_ref=registration,
            pid=os.getpid(),
            models=0,
            model_attempts=0,
            prediction_calls=0,
            response_evaluations=0,
            verified_responses=0,
            query_count=0,
            numerical_inner_steps=0,
            qp_solves=0,
            nnls_fit_attempts=0,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            actual_consumed_reference_count=0,
            control_qualified=False,
            automatic_retry=False,
        )
        native = dict(
            mj_forward_queries=0,
            numerical_inner_steps=0,
            max_jit_vs_python_goal_difference=0.0,
            jit_python_goal_differences=0,
        )
        dynamics = None
        entries = []
        s = self.source
        c = self.control
        try:
            result["environment_admission"] = restore_cv2_environment(
                self.guard["exact_probe_ref"], self.guard["runtime_source_refs"]
            )
            os.environ.pop("MUJOCO_GL", None)
            os.environ.pop("PYOPENGL_PLATFORM", None)
            result["model_attempts"] += 1
            dynamics = MujocoRobotDynamics(
                checked(self.model_reg["xml_ref"]),
                c["controller_site"],
                [f"robot0_joint{k}" for k in range(1, 8)],
                s["fixed_gripper_state"],
            )
            result["models"] += 1
            q0, v0 = (np.asarray(s[k]) for k in ["current_q", "current_velocity"])
            dynamics(q0, v0)
            interface = OSCInterface(*(c[k] for k in ["input_min", "input_max", "output_min", "output_max"]))
            limits, torque = (np.asarray(s[k]) for k in ["joint_limits", "torque_limits"])
            for index, case in enumerate(cases):
                command = np.asarray(case["commands"])
                result["prediction_calls"] += 1
                rollout = rollout_replay_order_trajectory(
                    q0,
                    v0,
                    command,
                    interface,
                    np.asarray(s["initial_joint"]),
                    np.asarray(s["kp"]),
                    np.asarray(s["kd"]),
                    dynamics,
                    torque[:, 0],
                    torque[:, 1],
                    latched_rotation_goal=np.asarray(s["latched_rotation_goal"]),
                )
                result["response_evaluations"] += 1
                result["numerical_inner_steps"] += 300
                entry = evaluate_repair_entry(
                    rollout,
                    command,
                    np.asarray(s["previous_command"])[:6],
                    interface,
                    limits[:, 0],
                    limits[:, 1],
                    s["desired_tcp"],
                    s["site_from_tcp"],
                    s["past_inner_error_rad"],
                    s["terminal_normals"],
                )
                publish_result(self.output / "production" / f"response_{index:02d}.json", entry)
                entries.append(entry)
                if index == 0:
                    require(all(r["exact"] for r in compare_values(expected_anchor, entry)), "Exact historical anchor")
                print(result_json(dict(production=index + 1, total=len(cases))), flush=True)
            result["repeat_exact"] = all(r["exact"] for r in compare_values(entries[0], entries[-1]))
            require(result["repeat_exact"], "Exact full initial repeat")
            result["model_attempts"] += 1
            reconstruct, compare_rollout, classify, check_audit, _, _ = case_engine(
                "prefix",
                s,
                self.model_reg,
                self.prefix_reg,
                native,
                self.refs,
                lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
                input_context=self.context,
            )
            result["models"] += 1
            exact = True
            for index, entry in enumerate(entries):
                command = np.asarray(entry["commands"])
                rebuilt, _ = reconstruct(command)
                metric = classify(command, rebuilt)
                check_full_candidate(
                    entry,
                    command,
                    rebuilt,
                    metric,
                    compare_rollout,
                    check_audit,
                    compare=lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
                )
                exact = exact and all(r["exact"] for r in compare_values(entry["rollout"], rebuilt))
                result["verified_responses"] += 1
                publish_result(
                    self.output / "independent" / f"response_{index:02d}.json", dict(rollout=rebuilt, metrics=metric)
                )
            require(exact, "Exact full independent response values")
            result["exact_native_responses"] = exact
            count = len(entries)
            require(dynamics.queries == native["mj_forward_queries"] == 1 + 312 * count, "Every actual query charged")
            result["total_queries"] = dynamics.queries + native["mj_forward_queries"]
            require(
                result["total_queries"] == detail["total_query_budget"]
                and result["numerical_inner_steps"] + native["numerical_inner_steps"]
                == detail["total_inner_step_budget"]
                and result["models"] == 2,
                "Exact registered model/query/inner-step costs",
            )
            result.update(complete=True, passed=True)
        except BaseException as error:
            result.update(error=repr(error), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            if dynamics is not None:
                result["query_count"] = dynamics.queries
            result["independent_counts"] = native
            try:
                self.context.finish()
                strict_native_environment()
                after = snapshot(REPO)
                result["source_after_ref"] = publish_result(self.output / "source_after.json", after)
                result["sources_unchanged"] = all(
                    self.before[k] == after[k] for k in ["sources", "current_map_ref", "driver_ref"]
                )
                result["passed"] = result["passed"] and result["sources_unchanged"]
            except BaseException as error:
                result.update(source_error=repr(error), passed=False)
            receipt = publish_result(self.output / "result.json", result)
            print(
                result_json(dict(result_ref=receipt, complete=result["complete"], passed=result["passed"])), flush=True
            )
        return result
