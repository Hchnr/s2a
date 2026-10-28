"""Evaluate14 fixed tail translations while retaining the original causal prefix."""

import argparse
import json
import os
import time

import numpy as np
from tail_translation_reserve_audit import audit_tail_translation_grid

from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result
from s2a.tail_translation_reserve import build_tail_translation_grid
from s2a.tail_translation_reserve_evidence import (
    CPU_ENVIRONMENT,
    PLAN,
    PRIOR_ANALYSIS_REF,
    PRIOR_CLOSURE_REF,
    PRIOR_PRODUCER_REF,
    REPO,
    SOURCES,
    admit_anchor,
    case_root,
    checked,
    finish,
    load_prerequisites,
    model_registration,
    read,
    reference,
    require,
)
from s2a.terminal_development_execution import CONTROL_REF


def prepare_case(source, prior, interface):
    """Bind the original anchor and all15 prior residuals before any new query."""
    answer = prior["answer"]
    anchor = admit_anchor(source, answer["geometry"]["anchor_commands"], interface)
    for key in ("input_min", "input_max"):
        require(np.array_equal(getattr(interface, key), answer[key]), "Original command bounds")
    previous = np.asarray(source["previous_command"], dtype=float)[:6]
    require(np.array_equal(previous, answer["previous_command"]), "Completed367 command identity")
    historical = answer["evaluated_candidates"][0]
    require(np.asarray(historical["commands"], dtype=float).tobytes() == anchor.tobytes(), "Original full anchor bytes")
    require(np.array_equal(source["terminal_normals"], [[0, 0, 0, 0, 0, 0, 1]]), "Original terminal normal e7")
    names = ("inner_lower", "inner_upper", "braking_lower", "braking_upper")
    offset = sum(np.asarray(historical["audit"]["groups"][key]["slack"]).size for key in names)
    require(offset == answer["terminal_row_index"] == 4368, "Original terminal row index")
    old_row = np.asarray(answer["fit"]["original_terminal_row"], dtype=float)
    require(old_row.shape == (72,) and np.isfinite(old_row).all(), "Full old uncorrected terminal row")
    entries, ledger, predictions = answer["evaluated_candidates"], answer["request_ledger"], answer["predictions"]
    require(
        len(entries) == len(ledger) == 17 and len(predictions) == 15, "All prior17 calls and15 held-out predictions"
    )
    base_terminal = float(historical["safety_slack"][offset])
    require(base_terminal == -historical["rollout"]["joint_velocity"][-1][6], "Historical terminal sign")
    residuals = []
    for index, prediction in enumerate(predictions, start=1):
        request, entry = ledger[index], entries[index]
        commands = np.asarray(entry["commands"], dtype=float)
        require(
            request["role"] == "grid"
            and request["response_index"] == request["request_index"] == prediction["request_index"] == index,
            "Every fixed old validation request remains in order",
        )
        require(request["command_key"] == commands.tobytes().hex(), "Old response command binding")
        terminal = float(entry["safety_slack"][offset])
        require(
            terminal == entry["audit"]["groups"]["terminal"]["slack"][0] == -entry["rollout"]["joint_velocity"][-1][6],
            "Every old terminal equals negative final joint7 velocity",
        )
        predicted = float(old_row @ (commands - anchor).ravel())
        observed = terminal - base_terminal
        residual = observed - predicted
        require(
            predicted == prediction["original_predicted_change"]
            and observed == prediction["observed_change"]
            and residual == prediction["original_residual"],
            "All prior original-precision residual algebra",
        )
        residuals.append(residual)
    analysis = read(PRIOR_ANALYSIS_REF)
    saved = analysis["analysis"]["points"]
    require(
        len(saved) == 15
        and all(row["request_index"] == i + 1 and row["old_residual"] == residuals[i] for i, row in enumerate(saved)),
        "Complete closed prior analysis cross-check",
    )
    error = max(abs(value) for value in residuals)
    require(np.isfinite(error) and error > 0, "Positive finite prior E17")
    grid = build_tail_translation_grid(anchor, previous, interface.input_min, interface.input_max)
    require(
        grid["admitted"] and grid["planned_request_count"] == 14 and grid["expected_unique_command_count"] == 13,
        "All12 unique tail candidates admitted before model construction",
    )
    audit_tail_translation_grid(grid)
    binding = dict(
        prior_producer_ref=PRIOR_PRODUCER_REF,
        prior_analysis_ref=PRIOR_ANALYSIS_REF,
        request_indices=list(range(1, 16)),
        original_residuals=residuals,
        validation_point_count=15,
        terminal_row_index=offset,
        historical_response_index=0,
        original_sqp_anchor_response_index=1819,
        prior_max_abs_residual=error,
        reserve_multiplier=10.0,
        safety_buffer=1e-8,
        reserve_target=10.0 * error,
        terminal_slack_target=1e-8 + 10.0 * error,
    )
    return grid, error, binding


def run_probe(
    source,
    grid,
    historical_anchor,
    prior_max_abs_residual,
    interface,
    dynamics,
    *,
    probe_fn=None,
    rollout_fn=rollout_native_goal_trajectory,
    progress=None,
):
    """Reset actual368 for every call; permit exactly one final anchor repeat."""
    if probe_fn is None:
        from s2a.tail_translation_reserve import probe_tail_translation_reserve

        probe_fn = probe_tail_translation_reserve
    anchor = admit_anchor(source, grid["anchor_commands"], interface)
    audit_tail_translation_grid(grid)
    expected = [anchor, *[np.asarray(row["commands"], dtype=float) for row in grid["grid_commands"]], anchor]
    require(grid["admitted"] and len(expected) == 14, "All12 commands admitted before the first query")
    require(len({row.tobytes() for row in expected}) == 13, "Only the final anchor repeats")
    for command in expected:
        admit_anchor(source, command, interface)
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
        require(len(evaluated) < 14, "Every actual call shares the fixed14 budget")
        require(commands.tobytes() == expected[len(evaluated)].tobytes(), "Exact preregistered call order")
        key = commands.tobytes()
        if not evaluated:
            require(key == anchor.tobytes(), "First billed response is exact anchor1819")
        elif key in seen:
            require(key == evaluated[0], "Only the final forced anchor may repeat; there is no grid cache")
            repeated = True
        # Preserve original command slacks and tolerance, without a warm-start clip.
        admit_anchor(source, commands, interface)
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
        interface,
        limits[:, 0].copy(),
        limits[:, 1].copy(),
        np.asarray(source["desired_tcp"]).copy(),
        np.asarray(source["site_from_tcp"]).copy(),
        past_inner_error_rad=np.asarray(source["past_inner_error_rad"]).copy(),
        terminal_normals=np.asarray(source["terminal_normals"]).copy(),
        grid=grid,
        prior_max_abs_residual=prior_max_abs_residual,
        historical_anchor=historical_anchor,
    )
    result["answer"] = answer
    n = len(evaluated)
    require(
        14
        == n
        == result["prediction_calls"]
        == result["response_evaluations"]
        == answer["response_evaluations"]
        == len(answer["evaluated_candidates"])
        == 14
        and repeated
        and evaluated[-1] == evaluated[0]
        and len(seen) == answer["unique_command_count"] == n - 1
        and answer["total_iterations"] == 0
        and answer["planned_request_count"] == 14
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
    source, source_reg, prior, closure, refs = load_prerequisites()
    root = case_root()
    require(not root.exists(), "No overwrite or automatic retry")
    control = read(source_reg["control_ref"])["contract"]
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    grid, prior_max_abs_residual, prior_error_binding = prepare_case(source, prior, interface)
    model_reg, model_proof = model_registration(source_reg, refs)
    finish(refs)
    if dry:
        print(json.dumps(dict(dry_run=True, passed=True, source_count=len(refs), responses=0, queries=0)), flush=True)
        return
    registration = publish_result(
        root / "registration.json",
        dict(
            experiment="tail_translation_reserve",
            asset_loading_proof=model_proof,
            prior_producer_ref=PRIOR_PRODUCER_REF,
            prior_closure_ref=PRIOR_CLOSURE_REF,
            prior_analysis_ref=PRIOR_ANALYSIS_REF,
            prior_closure=closure,
            grid=grid,
            prior_max_abs_residual=prior_max_abs_residual,
            prior_error_binding=prior_error_binding,
            plan_ref=reference(PLAN),
            seed=42,
            original_prefix_registration_ref=source["registration_ref"],
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            anchor_response_index=1819,
            signed_steps=[0.001, -0.001, 0.0005, -0.0005],
            endpoint=368,
            horizon=12,
            response_budget=14,
            unique_command_budget=13,
            planned_request_count=14,
            actual_consumed_reference_count=0,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="tail_translation_reserve",
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
        run_probe(
            source,
            grid,
            prior["answer"]["evaluated_candidates"][0],
            prior_max_abs_residual,
            interface,
            dynamics,
            progress=result,
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
