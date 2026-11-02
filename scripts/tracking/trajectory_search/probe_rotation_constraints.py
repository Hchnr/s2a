"""One post-hoc last-rotation direction under the original cap constraints."""

import copy
import os
import traceback
from pathlib import Path

import numpy as np

from s2a.controllers.osc import OSCInterface
from s2a.dynamics.mujoco_robot import MujocoRobotDynamics
from s2a.infrastructure.native_environment import restore_cv2_environment, strict_native_environment
from s2a.infrastructure.publication import publish_result, result_json
from s2a.research.trajectory_search.accepted_point_sqp_evidence import PREFIX_CHECK_REF
from s2a.research.trajectory_search.cap_distance_objective import CapDistanceObjective, cap_loss
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
from s2a.research.trajectory_search.state_vector_dual_recovery import objective_divisor, solve_recovered_epigraph_qp
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.tracking.trajectory_search.cap_distance_audit_policy import (
    CapDistanceAuditPolicy,
    projection_model,
)
from scripts.tracking.trajectory_search.cap_distance_audit_policy import cap_loss as independent_loss
from scripts.tracking.trajectory_search.check_replay_order_stencil import compare_values
from scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery import load_inputs
from scripts.tracking.trajectory_search.probe_tracking_v5_terminal_only_tail_repair import load_environment_guard
from scripts.tracking.trajectory_search.state_vector_dual_recovery_audit import _recovery_record, _scale_record
from scripts.tracking.trajectory_search.state_vector_gn_search_audit import _causal_update, _error, _periods, _pose
from scripts.verification.trajectory_search.accepted_point_sqp_math_audit import audit_qp_certificate
from scripts.verification.trajectory_search.check_tracking_v5_prefix_trust_control import check_full_candidate
from scripts.verification.trajectory_search.check_tracking_v5_saved_direction_backtracking import case_engine
from scripts.verification.trajectory_search.check_tracking_v5_tail_translation_reserve_v2 import semantic_compare

OUTPUT = REPO / "outputs/rotation_constraint_probe_20261102"
CONFIG = REPO / "configs/tracking/trajectory_search/rotation_constraint_probe_20261102.json"
PLAN = REPO / "plans/v0.0.1/tracking/trajectory_search/rotation_constraint_probe_20261102.md"
QA = REPO / "outputs/qa/rotation_constraint_probe_final_20261102/result.json"


def validate_spec(spec):
    expected = dict(
        seed=42,
        stage_index=58,
        anchor_index=85,
        target_group=23,
        alpha=1.0,
        radius=0.005,
        qp_budget=1,
        qp_iteration_budget=100,
        nnls_fit_budget=1,
        producer_response_budget=3,
        independent_response_budget=3,
        total_response_budget=6,
        total_inner_step_budget=1800,
        total_query_budget=1874,
        model_budget=2,
        timeout_seconds=600,
        automatic_retry=False,
        tool_policy="native_solve",
    )
    require(spec == expected, "Fixed one-direction diagnostic; no budget/target/constraint change")


def reconstruct_models(answer, stencil, stop):
    models = {k: np.asarray(stencil["matrices"][k]).copy() for k in ["Js", "Jq", "Je"]}
    sp, pp = _periods()
    ep = np.repeat(np.arange(12), 6)
    for stage in answer["stages"][:stop]:
        origin = answer["evaluated_candidates"][stage["anchor_before_index"]]
        source = stage["accepted_response_index"]
        if source is None and stage["rejected_update"] is not None:
            source = stage["rejected_update"]["source_response_index"]
        if source is None:
            continue
        observed = answer["evaluated_candidates"][source]
        step = (np.asarray(observed["commands"]) - np.asarray(origin["commands"])).ravel()
        for key, change, periods in [
            ("Js", np.asarray(observed["safety_slack"]) - np.asarray(origin["safety_slack"]), sp),
            ("Jq", _pose(observed) - _pose(origin), pp),
            ("Je", _error(observed) - _error(origin), ep),
        ]:
            models[key], _ = _causal_update(models[key], step, change, periods, answer["config"])
    return models


def rotation_problem(anchor, models, previous, lower, upper, ceiling, radius, config):
    problem = CapDistanceObjective.problem(
        anchor, models["Js"], models["Jq"], models["Je"], previous, lower, upper, ceiling, radius, "pose", config
    )
    r, J = problem["cap_model"]["residual"][-3:], problem["cap_model"]["jacobian"][-3:]
    problem.update(B=np.eye(72) + J.T @ J, g=J.T @ r, rotation_jacobian=J)
    return problem


def preflight(command, anchor, previous, input_min, input_max, lower, upper, radius, seen):
    step = command - anchor
    return dict(
        box=bool(np.all(command >= input_min - 1e-12) and np.all(command <= input_max + 1e-12)),
        rate=bool(np.all(abs(np.diff(np.vstack((previous, command)), axis=0)) <= 0.25 + 1e-12)),
        global_box=bool(np.all(command >= lower - 1e-12) and np.all(command <= upper + 1e-12)),
        trust=bool(np.max(abs(step)) <= radius + 1e-12),
        nonzero=bool(np.any(step != 0)),
        unique=command.tobytes() not in seen,
    )


def target_loss(entry):
    q = np.zeros(24)
    q[-1] = np.asarray(entry["normalized_squared_errors"]).ravel()[-1]
    return cap_loss(q)


def main():
    strict_native_environment()
    require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Single-thread CPU diagnostic")
    require(not OUTPUT.exists(), "Exclusive output; no retry")
    qa_ref = reference(QA)
    qa = read(qa_ref)
    require(
        qa["passed"] and qa["all_tests_passed"] and qa["source_unchanged"] and qa["skipped_count"] == 0,
        "Passing corresponding QA without skips",
    )
    qa_sources = read(qa["source_after_ref"])
    current_sources = snapshot(REPO)
    for key in ["sources", "current_map_ref", "driver_ref"]:
        require(current_sources[key] == qa_sources[key], "Source content matches QA: " + key)
    bundle, _, context = load_inputs()
    refs = context.runtime_refs
    remember(qa_ref, refs)
    remember(qa["source_after_ref"], refs)
    spec = read(reference(CONFIG))
    validate_spec(spec)
    result_ref = remember(reference(REPO / "outputs/cap_distance_gn_20261102/result.json"), refs)
    require(
        result_ref["sha256"] == "80f58cde84778d1e3b2f7ff3ca0b85c5414419da976a057710421d23f03012e9", "Closed cap result"
    )
    prior = read(result_ref)
    require(prior["complete"] and prior["passed"] and prior["sources_unchanged"], "Closed fully audited cap")
    cap = read(remember(prior["producer_final_ref"], refs))["answer"]
    stencil_ref = remember(reference(REPO / "outputs/replay_order_stencil_20261102/stencil.json"), refs)
    require(
        stencil_ref["sha256"] == "0d447082a50a2a8425c438a0a7cfe8a4a9742cb53bba445e015e49db6f6ecc3c", "Qualified stencil"
    )
    stencil = read(stencil_ref)
    stage = cap["stages"][spec["stage_index"]]
    require(
        stage["mode"] == "pose"
        and stage["qp"]["qualified"]
        and stage["anchor_before_index"] == spec["anchor_index"]
        and stage["radius_before"] == spec["radius"],
        "Fixed actual local problem",
    )
    anchor = cap["evaluated_candidates"][spec["anchor_index"]]
    models = reconstruct_models(cap, stencil, spec["stage_index"])
    source, source_reg, control = (bundle[k] for k in ["source", "source_registration", "control"])
    model_reg, proof = model_registration(source_reg, context, bundle)
    source = copy.deepcopy(source)
    control_record = read(remember(source_reg["control_ref"], refs))
    tool_record = read(remember(source_reg["tool_ref"], refs))
    source["site_from_tcp"] = np.linalg.solve(
        np.asarray(control_record["source_from_site"]), np.asarray(tool_record["source_from_tcp"])
    )
    previous = np.asarray(source["previous_command"])[:6]
    lower, upper = (np.asarray(cap[k]) for k in ["global_lower", "global_upper"])
    ceiling = np.asarray(cap["pose_ceiling"])
    cfg = cap["config"]
    x = np.asarray(anchor["commands"])
    args = (anchor, models, previous, lower, upper, ceiling, spec["radius"], cfg)
    problem = rotation_problem(*args)
    B, g, A, b = CapDistanceAuditPolicy.problem(
        x,
        np.asarray(anchor["safety_slack"]),
        _pose(anchor),
        _error(anchor),
        models["Js"],
        models["Jq"],
        models["Je"],
        previous,
        lower,
        upper,
        ceiling,
        spec["radius"],
        "pose",
        cfg,
    )
    np.testing.assert_array_equal(B, stage["qp"]["B"])
    audit_qp_certificate(stage["qp"], B, g, A, b)
    r, J = projection_model(_error(anchor), _pose(anchor), models["Je"])
    B = np.eye(72) + J[-3:].T @ J[-3:]
    g = J[-3:].T @ r[-3:]
    for name, value in dict(B=B, g=g, A=A, b=b).items():
        np.testing.assert_array_equal(problem[name], value)
    guard = load_environment_guard(refs)
    prefix = read(remember(PREFIX_CHECK_REF, refs))
    prefix_reg = read(remember(prefix["registration_ref"], refs))
    local_paths = [Path(__file__), CONFIG, PLAN]
    before = [reference(p) for p in local_paths]
    context.finish()
    registration = publish_result(
        OUTPUT / "registration.json",
        dict(
            experiment="rotation_constraint_probe",
            spec=spec,
            source_refs=before,
            input_refs=list(refs.values()),
            cap_result_ref=result_ref,
            stencil_ref=stencil_ref,
            asset_proof=proof,
            qa_ref=qa_ref,
            source_snapshot_ref=publish_result(OUTPUT / "source_before.json", current_sources),
            stage=spec["stage_index"],
            anchor=spec["anchor_index"],
            B=B,
            g=g,
            A=A,
            b=b,
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
        numerical_inner_steps=0,
        query_count=0,
        qp_solves=0,
        nnls_fit_attempts=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
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
    try:
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        result["model_attempts"] += 1
        dynamics = MujocoRobotDynamics(
            checked(model_reg["xml_ref"]),
            control["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            source["fixed_gripper_state"],
        )
        result["models"] += 1
        q0, v0 = (np.asarray(source[k]) for k in ["current_q", "current_velocity"])
        dynamics(q0, v0)
        interface = OSCInterface(*(control[k] for k in ["input_min", "input_max", "output_min", "output_max"]))
        torque = np.asarray(source["torque_limits"])
        limits = np.asarray(source["joint_limits"])

        def evaluate(command):
            require(result["prediction_calls"] < 3, "Fixed complete response budget")
            result["prediction_calls"] += 1
            rollout = rollout_replay_order_trajectory(
                q0,
                v0,
                command,
                interface,
                np.asarray(source["initial_joint"]),
                np.asarray(source["kp"]),
                np.asarray(source["kd"]),
                dynamics,
                torque[:, 0],
                torque[:, 1],
                latched_rotation_goal=np.asarray(source["latched_rotation_goal"]),
            )
            result["response_evaluations"] += 1
            result["numerical_inner_steps"] += 300
            e = evaluate_repair_entry(
                rollout,
                command,
                previous,
                interface,
                limits[:, 0],
                limits[:, 1],
                source["desired_tcp"],
                source["site_from_tcp"],
                source["past_inner_error_rad"],
                source["terminal_normals"],
            )
            entries.append(e)
            return e

        observed = evaluate(x)
        initial = compare_values(anchor, observed)
        require(all(v["exact"] for v in initial), "Exact closed anchor repeat")
        result["qp_solves"] += 1
        divisor = objective_divisor(problem, spec["radius"], "pose")
        qp = solve_recovered_epigraph_qp(
            B,
            g,
            A,
            b,
            problem["initial_t"],
            objective_divisor=divisor,
            dual_recovery_mode="pose",
            max_iterations=100,
            certificate_tolerance=1e-6,
        )
        result["qp"] = qp
        result["nnls_fit_attempts"] = qp["nnls_fit_attempts"]
        result["certificate"] = audit_qp_certificate(qp, B, g, A, b)
        result["scale_check"] = _scale_record(
            qp, B, g, A, b, divisor, "pose", lambda n, a, b: semantic_compare(n, a, b, 1e-10)
        )
        result["recovery_check"] = _recovery_record(
            qp, B, g, A, b, "pose", lambda n, a, b: semantic_compare(n, a, b, 1e-10)
        )
        result["trial_evaluated"] = False
        if qp["qualified"]:
            command = x + np.asarray(qp["point"])[:72].reshape(12, 6)
            step = (command - x).ravel()
            gates = preflight(
                command,
                x,
                previous,
                interface.input_min,
                interface.input_max,
                lower,
                upper,
                spec["radius"],
                {np.asarray(e["commands"]).tobytes() for e in cap["evaluated_candidates"]},
            )
            delta = J[-3:] @ step
            predicted = float(-(g @ step) - 0.5 * (delta @ delta))
            result.update(admission=gates, predicted_reduction=predicted)
            if all(gates.values()) and predicted > 0:
                trial = evaluate(command)
                result.update(
                    trial_evaluated=True,
                    actual_rotation_reduction=target_loss(observed) - target_loss(trial),
                    trial_original_feasible=trial["feasible"],
                )
        evaluate(x)
        result["repeat_exact"] = all(v["exact"] for v in compare_values(entries[0], entries[-1]))
        require(result["repeat_exact"], "Exact initial repeat")
        result["entries_ref"] = publish_result(OUTPUT / "entries.json", entries)
        result["model_attempts"] += 1
        reconstruct, compare_rollout, classify, check_audit, _, _ = case_engine(
            "prefix",
            source,
            model_reg,
            prefix_reg,
            native,
            refs,
            lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            input_context=context,
        )
        result["models"] += 1
        metrics = []
        exact = True
        for e in entries:
            rebuilt, _ = reconstruct(np.asarray(e["commands"]))
            metric = classify(np.asarray(e["commands"]), rebuilt)
            check_full_candidate(
                e,
                np.asarray(e["commands"]),
                rebuilt,
                metric,
                compare_rollout,
                check_audit,
                compare=lambda n, a, b, tolerance=1e-10: semantic_compare(n, a, b, tolerance),
            )
            exact = exact and all(v["exact"] for v in compare_values(e["rollout"], rebuilt))
            metrics.append(metric)
        result["exact_native_responses"] = exact
        require(exact, "Exact independent response values")
        if result["trial_evaluated"]:

            def independent_target(m):
                q = np.zeros(24)
                q[-1] = np.asarray(m["normalized_squared_errors"]).ravel()[-1]
                return independent_loss(q)

            native_reduction = independent_target(metrics[0]) - independent_target(metrics[1])
            semantic_compare("rotation_reduction", native_reduction, result["actual_rotation_reduction"], 1e-10)
            result["independent_rotation_reduction"] = native_reduction
        result["native_metrics_ref"] = publish_result(OUTPUT / "native_metrics.json", metrics)
        count = len(entries)
        require(
            count in (2, 3) and dynamics.queries == 1 + 312 * count and native["mj_forward_queries"] == 1 + 312 * count,
            "All actual queries charged",
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
            context.finish()
            strict_native_environment()
            after_sources = snapshot(REPO)
            result["source_after_ref"] = publish_result(OUTPUT / "source_after.json", after_sources)
            result["sources_unchanged"] = before == [reference(p) for p in local_paths] and all(
                current_sources[key] == after_sources[key] for key in ["sources", "current_map_ref", "driver_ref"]
            )
            result["passed"] = result["passed"] and result["sources_unchanged"]
        except BaseException as error:
            result.update(source_error=repr(error), passed=False)
        result_ref = publish_result(OUTPUT / "result.json", result)
        print(
            result_json(dict(result_ref=result_ref, complete=result["complete"], passed=result["passed"])), flush=True
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
