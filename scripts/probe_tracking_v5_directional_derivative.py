"""Measure the preregistered fixed direction grid without optimizing commands."""

import argparse
import json
import os
import time

import numpy as np
from probe_tracking_v5_prefix_trust_control import admit_warm

from s2a.directional_derivative_evidence import (
    CPU_ENVIRONMENT,
    PLAN,
    REPO,
    SOURCES,
    SQP_PRODUCER_REF,
    case_root,
    checked,
    compare_saved_stencil,
    finish,
    load_prerequisites,
    model_registration,
    read,
    reference,
    require,
    saved_stencil,
)
from s2a.directional_derivative_probe import STEP_SIZES, _repeat_comparison
from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result
from s2a.terminal_development_execution import CONTROL_REF


def run_probe(
    source,
    anchor,
    directions,
    interface,
    dynamics,
    *,
    probe_fn=None,
    rollout_fn=rollout_native_goal_trajectory,
    progress=None,
):
    """Reset actual368 for every call; permit exactly one final anchor repeat."""
    if probe_fn is None:
        from s2a.directional_derivative_probe import probe_directional_derivatives

        probe_fn = probe_directional_derivatives
    anchor = admit_warm(source, anchor, interface)
    directions = np.asarray(directions, dtype=float)
    require(directions.shape == (3, 12, 6) and np.isfinite(directions).all(), "Three complete saved directions")
    q, velocity = np.asarray(source["current_q"]).copy(), np.asarray(source["current_velocity"]).copy()
    latch = np.asarray(source["latched_rotation_goal"]).copy()
    limits, torque = np.asarray(source["joint_limits"]), np.asarray(source["torque_limits"])
    start_queries = dynamics.queries
    result = {} if progress is None else progress
    result.update(
        prediction_calls=0,
        response_evaluations=0,
        unique_command_count=0,
        total_iterations=0,
        numerical_inner_steps=0,
        query_count=0,
    )
    current = dynamics(q.copy(), velocity.copy())
    result["query_count"] = dynamics.queries - start_queries
    require(result["query_count"] == 1, "Exactly one current query")
    for key, value in current.items():
        np.testing.assert_allclose(value, source["current_quantities"][key], atol=1e-10, rtol=0)
    result["current_quantities"] = current
    evaluated, seen, repeated = [], set(), False

    def predict(commands):
        nonlocal repeated
        result["prediction_calls"] += 1
        commands = np.asarray(commands, dtype=float).copy()
        require(commands.shape == (12, 6) and np.isfinite(commands).all(), "Complete finite H12 predictor request")
        require(not repeated, "No predictor call after the final anchor repeat")
        require(len(evaluated) < 80, "Every actual call shares the fixed80 budget")
        key = commands.tobytes()
        if not evaluated:
            require(key == anchor.tobytes(), "First billed response is exact anchor1819")
        elif key in seen:
            require(key == evaluated[0], "Only the final anchor may repeat; grid cache owns all other duplicates")
            repeated = True
        # Enforce command admission independently of the core before an expensive call.
        increments = np.diff(np.vstack((np.asarray(source["previous_command"])[:6], commands)), axis=0)
        require(
            np.all(commands >= interface.input_min - 1e-12)
            and np.all(commands <= interface.input_max + 1e-12)
            and np.all(np.abs(increments) <= 0.25 + 1e-12),
            "Original complete command box and rate admission; no clipping",
        )
        before = dynamics.queries
        try:
            rollout = rollout_fn(
                q.copy(),
                velocity.copy(),
                commands.copy(),
                interface,
                np.asarray(source["initial_joint"]).copy(),
                np.asarray(source["kp"]).copy(),
                np.asarray(source["kd"]).copy(),
                dynamics,
                torque[:, 0].copy(),
                torque[:, 1].copy(),
                latched_rotation_goal=latch.copy(),
            )
        finally:
            result["query_count"] = dynamics.queries - start_queries
        result["response_evaluations"] += 1
        result["numerical_inner_steps"] += 300
        require(dynamics.queries - before == 312, "Every full H12 call bills300 inner plus12 endpoint queries")
        seen.add(key)
        evaluated.append(key)
        result["unique_command_count"] = len(seen)
        if len(evaluated) == 1 or len(evaluated) % 10 == 0:
            print(json.dumps(dict(responses=len(evaluated), unique_commands=len(seen))), flush=True)
        return rollout

    answer = probe_fn(
        predict,
        np.asarray(source["previous_command"])[:6].copy(),
        anchor.copy(),
        directions.copy(),
        interface,
        limits[:, 0].copy(),
        limits[:, 1].copy(),
        np.asarray(source["desired_tcp"]).copy(),
        np.asarray(source["site_from_tcp"]).copy(),
        past_inner_error_rad=np.asarray(source["past_inner_error_rad"]).copy(),
        terminal_normals=np.asarray(source["terminal_normals"]).copy(),
        source_stages=(24, 25, 26),
        step_sizes=STEP_SIZES,
        max_responses=80,
    )
    result["answer"] = answer
    n = len(evaluated)
    require(
        2
        <= n
        == result["prediction_calls"]
        == result["response_evaluations"]
        == answer["response_evaluations"]
        == len(answer["evaluated_candidates"])
        <= 80
        and repeated
        and evaluated[-1] == evaluated[0]
        and len(seen) == answer["unique_command_count"] == n - 1
        and answer["total_iterations"] == 0
        and answer["planned_request_count"] == 80
        and result["numerical_inner_steps"] == n * 300
        and result["query_count"] == 1 + n * 312,
        "All full calls including the forced final repeat are retained and billed once",
    )
    require(
        evaluated == [np.asarray(row["commands"], dtype=float).tobytes() for row in answer["evaluated_candidates"]],
        "Exact original call order and bytes",
    )
    result["witness_indices"] = [i for i, row in enumerate(answer["evaluated_candidates"]) if row["feasible"]]
    result["witness_found"] = bool(result["witness_indices"])
    return result


def main(dry=False):
    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU")
    source, source_reg, sqp, closure, refs = load_prerequisites()
    root = case_root()
    require(not root.exists(), "No overwrite or automatic retry")
    control = read(source_reg["control_ref"])["contract"]
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    anchor, directions, Js, Jq = saved_stencil(sqp["answer"])
    anchor = admit_warm(source, anchor, interface)
    model_reg, model_proof = model_registration(source_reg, refs)
    finish(refs)
    if dry:
        print(json.dumps(dict(dry_run=True, passed=True, source_count=len(refs), responses=0, queries=0)), flush=True)
        return
    registration = publish_result(
        root / "registration.json",
        dict(
            experiment="directional_derivative_consistency",
            asset_loading_proof=model_proof,
            sqp_producer_ref=SQP_PRODUCER_REF,
            sqp_closure=closure,
            plan_ref=reference(PLAN),
            seed=42,
            original_prefix_registration_ref=source["registration_ref"],
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            anchor_response_index=1819,
            source_stages=[24, 25, 26],
            step_sizes=STEP_SIZES,
            endpoint=368,
            horizon=12,
            response_budget=80,
            unique_command_budget=79,
            planned_request_count=80,
            actual_consumed_reference_count=0,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="directional_derivative_consistency",
        registration_ref=registration,
        asset_loading_proof=model_proof,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        control_qualified=False,
        retiming_qualified=False,
        task_qualified=False,
        automatic_retry=False,
    )
    started = time.monotonic()
    try:
        guard = read(CONTROL_REF)["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        dynamics = MujocoRobotDynamics(
            checked(model_reg["xml_ref"]),
            control["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            source["fixed_gripper_state"],
        )
        require(dynamics.model.opt.timestep == 0.002, "Original2ms XML")
        with np.load(checked(source_reg["states_ref"]), allow_pickle=False) as archive:
            state = archive["sim_states"][368].copy()
        np.testing.assert_array_equal(source["current_q"], state[1 + dynamics.qids])
        np.testing.assert_array_equal(source["current_velocity"], state[1 + dynamics.model.nq + dynamics.dofs])
        np.testing.assert_array_equal(source["joint_limits"], dynamics.model.jnt_range[dynamics.joints])
        for name, fixed in source["fixed_gripper_state"].items():
            jid = dynamics.mj.mj_name2id(dynamics.model, dynamics.mj.mjtObj.mjOBJ_JOINT, name)
            require(jid >= 0, "Actual368 finger coordinate")
            np.testing.assert_array_equal(
                fixed,
                [
                    state[1 + dynamics.model.jnt_qposadr[jid]],
                    state[1 + dynamics.model.nq + dynamics.model.jnt_dofadr[jid]],
                ],
            )
        run_probe(source, anchor, directions, interface, dynamics, progress=result)
        result["saved_stencil_comparison"] = compare_saved_stencil(result["answer"], Js, Jq)
        # This comparison uses the historical trace/anchor record, not the separate saved ray rollout.
        old = dict(sqp["answer"]["evaluated_candidates"][1819])
        new = result["answer"]["evaluated_candidates"][0]
        anchors = [row for row in sqp["answer"]["accepted_anchors"] if row["response_index"] == 1819]
        require(len(anchors) == 1, "One saved accepted anchor1819 audit")
        old["audit"] = anchors[0]["audit"]
        projection = {
            key: new[key] for key in ("commands", "tcp_errors", "safety_slack", "prefix_length", "feasible", "audit")
        }
        projection.update(safety_min=float(np.min(new["safety_slack"])), joint_feasible=new["audit"]["feasible"])
        comparison = _repeat_comparison(old, projection, 0)
        del comparison["initial_response_index"], comparison["repeated_response_index"]
        result["historical_anchor_comparison"] = dict(
            scope="Complete archived trace and accepted-anchor audit versus new response0; no old full rollout",
            historical_response_index=1819,
            new_response_index=0,
            comparison=comparison,
        )
        finish(refs)
        strict_native_environment()
        result.update(
            complete=True, passed=True, source_refs_unchanged=list(refs.values()), elapsed_s=time.monotonic() - started
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(root / "verified.json", result)
    print(
        json.dumps(
            dict(
                complete=True,
                passed=True,
                responses=result["response_evaluations"],
                witness_found=result["witness_found"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry)
