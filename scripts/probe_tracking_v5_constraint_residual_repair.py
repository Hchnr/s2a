"""One minimum-norm correction of an archived, rejected full-H12 direction."""

import argparse
import json
import os
import time

import numpy as np
from probe_tracking_v5_terminal_only_tail_repair import load_environment_guard

from s2a.constraint_residual_repair import (
    build_repair_problem,
    classify_repair_entry,
    evaluate_repair_entry,
    fixed_repair_commands,
    solve_repair,
)
from s2a.directional_derivative_probe import _repeat_comparison
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
from s2a.terminal_only_tail_repair import ENTRY_FIELDS

ROOT = REPO / "outputs/tracking_v5_constraint_residual_repair_20261029"
INPUT = REPO / "outputs/constraint_residual_repair_inputs_20261029/bundle.json"
BUNDLE_SHA = "eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023"
PLAN = REPO / "plans/v0.0.1/constraint_residual_repair_20261029.md"
SOURCES = (
    "src/s2a/constraint_residual_repair.py",
    "scripts/probe_tracking_v5_constraint_residual_repair.py",
    "scripts/check_tracking_v5_constraint_residual_repair.py",
    "tests/test_constraint_residual_repair.py",
    "tests/test_constraint_residual_repair_check.py",
    "tests/test_constraint_residual_repair_execution.py",
    "plans/v0.0.1/constraint_residual_repair_20261029.md",
)


def load_inputs():
    bundle = read(dict(path=str(INPUT), sha256=BUNDLE_SHA))
    require(
        bundle["base_response_index"] == 3557
        and bundle["trial_response_index"] == 3639
        and bundle["probe_response_indices"] == list(range(3558, 3630))
        and bundle["phase"]["stage_index"] == 53
        and bundle["phase"]["radius"] == 0.005,
        "One preregistered historical direction and its complete original stencil",
    )
    refs = {ref["path"]: ref for ref in bundle["input_refs"]}
    for ref in (bundle["closure_ref"], bundle["producer_ref"], bundle["source_bundle_ref"], reference(INPUT)):
        remember(ref, refs)
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    # The environment helper and full-entry schema are reused without changing
    # their previously closed source versions.
    for path in (
        "scripts/probe_tracking_v5_terminal_only_tail_repair.py",
        "src/s2a/terminal_only_tail_repair.py",
    ):
        remember(reference(REPO / path), refs)
    finish(refs)
    return bundle, refs


def prepare_problem(bundle):
    source, control, phase = (bundle[key] for key in ("source", "control", "phase"))
    return build_repair_problem(
        bundle["base"],
        bundle["trial"],
        bundle["probes"],
        bundle["nominal_deltas"],
        np.asarray(source["previous_command"])[:6],
        control["input_min"],
        control["input_max"],
        phase["stage_lower_command"],
        phase["stage_upper_command"],
    )


def run_responses(source, trial, rows, threshold, interface, dynamics, result, *, rollout_fn=None):
    """All eleven complete calls use copies of the same actual368 causal state."""
    if rollout_fn is None:
        rollout_fn = rollout_native_goal_trajectory
    anchor = admit_anchor(source, trial["commands"], interface)
    require(len(rows) == 9, "One fixed nine-alpha correction sequence")
    commands = [anchor, *[admit_anchor(source, row["commands"], interface) for row in rows], anchor]
    require(len({command.tobytes() for command in commands}) == 10, "Only the final forced repeat may duplicate")
    q, velocity, latch = (
        np.asarray(source[key]).copy() for key in ("current_q", "current_velocity", "latched_rotation_goal")
    )
    limits, torque = (np.asarray(source[key]) for key in ("joint_limits", "torque_limits"))
    start_queries = dynamics.queries
    answer = dict(
        evaluated_candidates=[],
        request_ledger=[],
        classifications=[],
        selected_response_index=None,
        unique_command_count=0,
        response_evaluations=0,
        historical_exact=False,
        repeat_exact=False,
        witness_indices=[],
        witness_found=False,
        control_qualified=False,
    )
    result["answer"] = answer
    try:
        current = dynamics(q.copy(), velocity.copy())
    finally:
        result["query_count"] = dynamics.queries - start_queries
    require(result["query_count"] == 1, "Exactly one current-state query")
    for key, value in current.items():
        np.testing.assert_allclose(value, source["current_quantities"][key], atol=1e-10, rtol=0)
    result["current_quantities"] = current
    seen = set()
    for index, command in enumerate(commands):
        result["prediction_calls"] += 1
        before = dynamics.queries
        request = dict(
            request_index=index,
            role="trial" if index == 0 else "trial_repeat" if index == 10 else "correction",
            alpha=None if index in (0, 10) else rows[index - 1]["alpha"],
            commands=command.copy(),
            command_key=command.tobytes().hex(),
            completed=False,
        )
        answer["request_ledger"].append(request)
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
        request["completed"] = True
        require(dynamics.queries - before == 312, "Full H12 costs300 inner and12 endpoint queries")
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
        answer["evaluated_candidates"].append(entry)
        answer["classifications"].append(classify_repair_entry(entry, threshold))
        seen.add(command.tobytes())
        answer["response_evaluations"] = result["response_evaluations"]
        answer["unique_command_count"] = result["unique_command_count"] = len(seen)
        print(json.dumps(dict(responses=len(answer["evaluated_candidates"]), unique_commands=len(seen))), flush=True)
    entries = answer["evaluated_candidates"]
    require(set(ENTRY_FIELDS) <= set(trial), "Complete historical numeric entry")
    history = _repeat_comparison({key: trial[key] for key in ENTRY_FIELDS}, entries[0], 0)
    repeat = _repeat_comparison(entries[0], entries[-1], 10)
    answer.update(
        historical_comparison=history,
        repeatability=repeat,
        historical_exact=history["exact"],
        repeat_exact=repeat["exact"],
    )
    if history["exact"] and repeat["exact"]:
        answer["selected_response_index"] = next(
            (i for i in range(1, 10) if answer["classifications"][i]["repair_eligible"]), None
        )
    answer["witness_indices"] = [i for i, entry in enumerate(entries) if entry["feasible"]]
    answer["witness_found"] = bool(answer["witness_indices"])
    result["witness_indices"], result["witness_found"] = answer["witness_indices"], answer["witness_found"]
    require(
        result["prediction_calls"] == result["response_evaluations"] == len(entries) == 11
        and result["numerical_inner_steps"] == 3300
        and result["query_count"] == 3433,
        "All eleven actual complete responses including repeat are billed",
    )
    return answer


def main(dry=False):
    strict_native_environment()
    require(all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()), "Fixed single-thread CPU")
    bundle, refs = load_inputs()
    source, source_reg, control = (bundle[key] for key in ("source", "source_registration", "control"))
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    problem = prepare_problem(bundle)
    model_reg, proof = model_registration(source_reg, refs)
    guard = load_environment_guard(refs)
    if dry:
        print(
            json.dumps(dict(passed=True, dry=True, qp_solves=0, responses=0, queries=0, source_count=len(refs))),
            flush=True,
        )
        return
    require(not ROOT.exists(), "Exclusive output; no overwrite or automatic retry")
    registration = publish_result(
        ROOT / "registration.json",
        dict(
            experiment="constraint_residual_repair",
            input_ref=reference(INPUT),
            plan_ref=reference(PLAN),
            source_refs=[reference(REPO / p) for p in SOURCES],
            input_refs=list(refs.values()),
            base_response_index=3557,
            trial_response_index=3639,
            response_budget=11,
            qp_budget=1,
            qp_iteration_budget=100,
            seed=42,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="constraint_residual_repair",
        registration_ref=registration,
        asset_loading_proof=proof,
        problem=problem,
        qp=None,
        command_plan=None,
        answer=None,
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
        witness_found=False,
    )
    started = time.monotonic()
    try:
        result["qp_solves"] = 1
        qp = result["qp"] = solve_repair(problem)
        command_plan = result["command_plan"] = fixed_repair_commands(problem, qp)
        # A failed local correction is a scientific result, with no new queries.
        if command_plan["admitted"]:
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
            run_responses(
                source, bundle["trial"], command_plan["commands"], problem["threshold"], interface, dynamics, result
            )
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["partial_response_queries"] = max(
            0, result["query_count"] - int(result["query_count"] > 0) - 312 * result["response_evaluations"]
        )
        publish_result(ROOT / "verified.json", result)
    print(
        json.dumps(
            dict(
                passed=True,
                qp_qualified=qp["qualified"],
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
