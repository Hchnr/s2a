"""Search complete H12 trajectories with recurring restoration and measured ratios."""

import argparse
import json
import os
import time

import numpy as np
from probe_tracking_v5_constraint_residual_repair import load_inputs as load_prior_inputs
from probe_tracking_v5_constraint_residual_repair import prepare_problem
from probe_tracking_v5_terminal_only_tail_repair import load_environment_guard

from s2a.constraint_residual_repair import evaluate_repair_entry
from s2a.interior_trajectory_search import SearchEvaluationError, search
from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result
from s2a.restored_anchor_sqp_evidence import (
    CPU_ENVIRONMENT,
    REPO,
    checked,
    finish,
    model_registration,
    read,
    reference,
    remember,
    require,
)
from s2a.tail_translation_reserve_evidence import admit_anchor

ROOT = REPO / "outputs/tracking_v5_interior_trajectory_search_20261029"
PLAN = REPO / "plans/v0.0.1/interior_trajectory_search_20261029.md"
CONFIG = REPO / "configs/interior_trajectory_search_20261029.json"
PRIOR_ANALYSIS = dict(
    path=str(REPO / "outputs/constraint_residual_repair_analysis_20261029/result.json"),
    sha256="2f058ce0d7f638e017e7be3120b6e9fc2489535de60940586390c2118b122d01",
)
SOURCES = (
    "src/s2a/interior_trajectory_search.py",
    "scripts/interior_trajectory_search_audit.py",
    "scripts/probe_tracking_v5_interior_trajectory_search.py",
    "scripts/check_tracking_v5_interior_trajectory_search.py",
    "tests/test_interior_trajectory_search.py",
    "tests/test_interior_trajectory_search_audit.py",
    "tests/test_interior_trajectory_search_execution.py",
    "configs/interior_trajectory_search_20261029.json",
    "plans/v0.0.1/interior_trajectory_search_20261029.md",
)


def load_inputs():
    bundle, refs = load_prior_inputs()
    previous = read(remember(PRIOR_ANALYSIS, refs))
    require(
        previous["complete"] is previous["passed"] is previous["sources_unchanged"] is True
        and previous["scientific_repair_accepted"] is False
        and previous["total_response_calls"] == 22,
        "Previous fixed correction is closed without changing its failed selection",
    )
    for ref in previous["input_refs"].values():
        remember(ref, refs)
    config = read(remember(reference(CONFIG), refs))
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    finish(refs)
    return bundle, config, refs


def initial_model(bundle):
    """Reconstruct the old 3557 stencil without new finite-difference queries."""
    problem = prepare_problem(bundle)
    return problem["Js"].copy(), problem["Jq"].copy()


def run_search(source, historical, Js, Jq, config, interface, dynamics, result, *, search_fn=None, rollout_fn=None):
    """Every callback starts from the same causal actual368 state, never a rollout state."""
    if search_fn is None:
        search_fn = search
    if rollout_fn is None:
        rollout_fn = rollout_native_goal_trajectory
    admit_anchor(source, historical["commands"], interface)
    q, velocity, latch = (
        np.asarray(source[key]).copy() for key in ("current_q", "current_velocity", "latched_rotation_goal")
    )
    limits, torque = (np.asarray(source[key]) for key in ("joint_limits", "torque_limits"))
    start_queries = dynamics.queries
    try:
        current = dynamics(q.copy(), velocity.copy())
    finally:
        result["query_count"] = dynamics.queries - start_queries
    require(result["query_count"] == 1, "One initial-state query")
    for key, value in current.items():
        np.testing.assert_allclose(value, source["current_quantities"][key], atol=1e-10, rtol=0)
    result["current_quantities"] = current
    seen = set()

    def evaluate(commands):
        require(result["prediction_calls"] < config["max_responses"], "One global response budget including repeat")
        command = admit_anchor(source, commands, interface)
        result["prediction_calls"] += 1
        before = dynamics.queries
        try:
            rollout = rollout_fn(
                q.copy(),
                velocity.copy(),
                command.copy(),
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
        require(dynamics.queries - before == 312, "Every complete H12 costs300 inner and12 endpoint queries")
        seen.add(command.tobytes())
        result["unique_command_count"] = len(seen)
        entry = evaluate_repair_entry(
            rollout,
            command,
            np.asarray(source["previous_command"])[:6],
            interface,
            limits[:, 0],
            limits[:, 1],
            source["desired_tcp"],
            source["site_from_tcp"],
            source["past_inner_error_rad"],
            source["terminal_normals"],
        )
        print(
            json.dumps(
                dict(
                    responses=result["response_evaluations"],
                    prefix=entry["prefix_length"],
                    maximum_error=entry["maximum_normalized_squared_error"],
                    safety_min=float(np.min(entry["safety_slack"])),
                )
            ),
            flush=True,
        )
        return entry

    try:
        answer = search_fn(
            evaluate,
            historical,
            Js.copy(),
            Jq.copy(),
            np.asarray(source["previous_command"])[:6].copy(),
            np.asarray(interface.input_min).copy(),
            np.asarray(interface.input_max).copy(),
            config,
        )
    except SearchEvaluationError as error:
        result["answer"] = error.answer
        raise
    result["answer"] = answer
    count = result["response_evaluations"]
    require(
        count == result["prediction_calls"] == answer["response_evaluations"] == len(answer["evaluated_candidates"])
        and result["unique_command_count"] == answer["unique_command_count"]
        and result["numerical_inner_steps"] == 300 * count
        and result["query_count"] == 1 + 312 * count,
        "Complete response ledger and actual costs agree, including the mandatory repeat",
    )
    result["witness_indices"] = answer["witness_indices"]
    result["qualified_witness_indices"] = answer["qualified_witness_indices"]
    result["witness_found"] = bool(answer["witness_indices"])
    return answer


def main(dry=False):
    strict_native_environment()
    require(
        all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Single-thread CPU environment"
    )
    bundle, config, refs = load_inputs()
    source, source_reg, control = (bundle[key] for key in ("source", "source_registration", "control"))
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    Js, Jq = initial_model(bundle)
    model_reg, proof = model_registration(source_reg, refs)
    guard = load_environment_guard(refs)
    if dry:
        print(
            json.dumps(dict(passed=True, dry=True, qp_solves=0, responses=0, queries=0, source_count=len(refs))),
            flush=True,
        )
        return
    require(not ROOT.exists(), "Exclusive experiment output; no overwrite or automatic retry")
    registration = publish_result(
        ROOT / "registration.json",
        dict(
            experiment="interior_trajectory_search",
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            plan_ref=reference(PLAN),
            config_ref=reference(CONFIG),
            config=config,
            historical_response_index=3557,
            response_budget=config["max_responses"],
            qp_budget=config["max_stages"],
            seed=config["seed"],
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="interior_trajectory_search",
        registration_ref=registration,
        asset_loading_proof=proof,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        control_qualified=False,
        automatic_retry=False,
        prediction_calls=0,
        response_evaluations=0,
        unique_command_count=0,
        numerical_inner_steps=0,
        query_count=0,
        qp_solves=0,
        model_construction_attempts=0,
        model_constructions_completed=0,
        witness_indices=[],
        qualified_witness_indices=[],
        witness_found=False,
        answer=None,
    )
    started = time.monotonic()
    try:
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        result["model_construction_attempts"] += 1
        dynamics = MujocoRobotDynamics(
            checked(model_reg["xml_ref"]),
            control["controller_site"],
            [f"robot0_joint{k}" for k in range(1, 8)],
            source["fixed_gripper_state"],
        )
        result["model_constructions_completed"] += 1
        require(dynamics.model.opt.timestep == 0.002, "Original2ms model")
        with np.load(checked(source_reg["states_ref"]), allow_pickle=False) as archive:
            state = archive["sim_states"][368].copy()
        np.testing.assert_array_equal(source["current_q"], state[1 + dynamics.qids])
        np.testing.assert_array_equal(source["current_velocity"], state[1 + dynamics.model.nq + dynamics.dofs])
        np.testing.assert_array_equal(source["joint_limits"], dynamics.model.jnt_range[dynamics.joints])
        run_search(source, bundle["base"], Js, Jq, config, interface, dynamics, result)
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        if result["answer"] is not None:
            result["qp_solves"] = len(result["answer"]["stages"])
        result["partial_response_queries"] = max(
            0, result["query_count"] - int(result["query_count"] > 0) - 312 * result["response_evaluations"]
        )
        publish_result(ROOT / "verified.json", result)
    print(
        json.dumps(
            dict(
                passed=True,
                responses=result["response_evaluations"],
                witness_found=result["witness_found"],
                termination=result["answer"]["termination_reason"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry)
