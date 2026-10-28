"""Evaluate17 fixed holdout commands for the frozen terminal-row secant correction."""

import argparse
import json
import os
import time

import numpy as np

from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface
from s2a.preview_result_publication import publish_result
from s2a.terminal_development_execution import CONTROL_REF
from s2a.terminal_secant_evidence import (
    CPU_ENVIRONMENT,
    DIAGNOSTIC_CLOSURE_REF,
    DIAGNOSTIC_REF,
    PLAN,
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
from s2a.terminal_secant_probe import STEP_SIZES, build_terminal_secant_geometry, fit_terminal_secant_row


def prepare_case(source, diagnostic, interface):
    """Bind two archived training records and preflight all future commands."""
    answer = diagnostic["answer"]
    anchor = admit_anchor(source, answer["anchor_commands"], interface)
    for key in ("input_min", "input_max"):
        require(np.array_equal(getattr(interface, key), answer[key]), "Original unchanged command bounds")
    previous = np.asarray(source["previous_command"], dtype=float)[:6]
    require(np.array_equal(previous, answer["previous_command"]), "Completed367 command identity")
    old = answer["evaluated_candidates"][0]
    require(
        np.asarray(old["commands"], dtype=float).tobytes() == anchor.tobytes(), "Original full anchor command bytes"
    )
    normals = np.asarray(source["terminal_normals"], dtype=float)
    require(
        normals.shape == (1, 7) and np.array_equal(normals[0], [0, 0, 0, 0, 0, 0, 1]), "Original terminal normal e7"
    )
    names = ("inner_lower", "inner_upper", "braking_lower", "braking_upper")
    offset = sum(np.asarray(old["audit"]["groups"][key]["slack"]).size for key in names)
    require(offset == 4368 and len(old["safety_slack"]) == 4369, "Original terminal row layout")
    row = np.asarray(diagnostic["saved_stencil_comparison"]["jacobian_safety_slack"], dtype=float)[offset]
    records, request_indices, response_indices = [], [], []
    for step in (0.001, 0.0005):
        requests = [
            r
            for r in answer["request_ledger"]
            if r["role"] == "grid" and r["source_stage"] == 24 and r["step"] == step and r["sign"] == 1
        ]
        require(len(requests) == 1, "Exactly one frozen training request at each registered scale")
        request = requests[0]
        require(request["outcome"] == "computed" and request["charged"] is True, "Actual completed training response")
        record = answer["evaluated_candidates"][request["response_index"]]
        require(
            np.asarray(record["commands"], dtype=float).tobytes().hex() == request["command_key"],
            "Training response command binding",
        )
        records.append(record)
        request_indices.append(request["request_index"])
        response_indices.append(request["response_index"])
    for record in [old, *records]:
        require(
            record["safety_slack"][offset]
            == record["audit"]["groups"]["terminal"]["slack"][0]
            == -record["rollout"]["joint_velocity"][-1][6],
            "Terminal row is negative final joint7 velocity",
        )
    geometry = build_terminal_secant_geometry(
        anchor, previous, interface.input_min, interface.input_max, answer["unit_directions"]
    )
    require(
        geometry["admitted"]
        and geometry["planned_request_count"] == 17
        and geometry["expected_unique_command_count"] == 16,
        "All15 fixed unique probes admitted before model construction",
    )
    fit = fit_terminal_secant_row(
        anchor,
        answer["unit_directions"][0],
        row,
        old["safety_slack"][offset],
        [r["commands"] for r in records],
        [r["safety_slack"][offset] for r in records],
    )
    source_binding = dict(
        diagnostic_ref=DIAGNOSTIC_REF,
        source_stage=24,
        sign=1,
        step_sizes=[0.001, 0.0005],
        request_indices=request_indices,
        response_indices=response_indices,
        terminal_index=offset,
        diagnostic_anchor_response_index=0,
        original_sqp_anchor_response_index=1819,
    )
    return geometry, fit, source_binding


def run_probe(
    source,
    geometry,
    fit,
    historical_anchor,
    interface,
    dynamics,
    *,
    probe_fn=None,
    rollout_fn=rollout_native_goal_trajectory,
    progress=None,
):
    """Reset actual368 for every call; permit exactly one final anchor repeat."""
    if probe_fn is None:
        from s2a.terminal_secant_probe import probe_terminal_secant

        probe_fn = probe_terminal_secant
    anchor = admit_anchor(source, geometry["anchor_commands"], interface)
    expected = [anchor, *[np.asarray(row["commands"], dtype=float) for row in geometry["grid_commands"]], anchor]
    require(geometry["admitted"] and len(expected) == 17, "All15 commands admitted before the first query")
    require(len({row.tobytes() for row in expected}) == 16, "Only the final anchor repeats")
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
        require(len(evaluated) < 17, "Every actual call shares the fixed17 budget")
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
        geometry=geometry,
        fit=fit,
        historical_anchor=historical_anchor,
    )
    result["answer"] = answer
    n = len(evaluated)
    require(
        17
        == n
        == result["prediction_calls"]
        == result["response_evaluations"]
        == answer["response_evaluations"]
        == len(answer["evaluated_candidates"])
        == 17
        and repeated
        and evaluated[-1] == evaluated[0]
        and len(seen) == answer["unique_command_count"] == n - 1
        and answer["total_iterations"] == 0
        and answer["planned_request_count"] == 17
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
    source, source_reg, diagnostic, closure, refs = load_prerequisites()
    root = case_root()
    require(not root.exists(), "No overwrite or automatic retry")
    control = read(source_reg["control_ref"])["contract"]
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    geometry, fit, fit_source = prepare_case(source, diagnostic, interface)
    model_reg, model_proof = model_registration(source_reg, refs)
    finish(refs)
    if dry:
        print(json.dumps(dict(dry_run=True, passed=True, source_count=len(refs), responses=0, queries=0)), flush=True)
        return
    registration = publish_result(
        root / "registration.json",
        dict(
            experiment="terminal_secant_probe",
            asset_loading_proof=model_proof,
            diagnostic_ref=DIAGNOSTIC_REF,
            diagnostic_closure_ref=DIAGNOSTIC_CLOSURE_REF,
            prior_closure=closure,
            geometry=geometry,
            fit=fit,
            fit_source=fit_source,
            plan_ref=reference(PLAN),
            seed=42,
            original_prefix_registration_ref=source["registration_ref"],
            source_refs=[reference(REPO / path) for path in SOURCES],
            input_refs=list(refs.values()),
            anchor_response_index=1819,
            step_sizes=STEP_SIZES,
            endpoint=368,
            horizon=12,
            response_budget=17,
            unique_command_budget=16,
            planned_request_count=17,
            actual_consumed_reference_count=0,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        experiment="terminal_secant_probe",
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
            source, geometry, fit, diagnostic["answer"]["evaluated_candidates"][0], interface, dynamics, progress=result
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
