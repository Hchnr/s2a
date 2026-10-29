"""Evaluate the preregistered six scales on every saved main-subset direction."""

import argparse
import json
import os
import time

import numpy as np

from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface
from s2a.paired_reference_phase import score_phase_response
from s2a.preview_result_publication import publish_result
from s2a.saved_direction_backtracking import CASE_ORDER, build_batch, score_backtracking_response
from s2a.saved_direction_backtracking_evidence import (
    CPU_ENVIRONMENT,
    DIAGNOSTIC_REF,
    PLAN,
    REPO,
    ROOT,
    SOURCES,
    checked,
    finish,
    load_prerequisites,
    read,
    reference,
    require,
)
from s2a.terminal_development_execution import CONTROL_REF

EXPECTED_COUNTS = dict(
    prefix=dict(response_evaluations=60, numerical_inner_steps=18000, query_count=18721),
    early=dict(response_evaluations=30, numerical_inner_steps=9750, query_count=10141),
    late=dict(response_evaluations=6, numerical_inner_steps=1950, query_count=2029),
)


def compare_saved_endpoint(request, classification, trace):
    """Check exactly the fields archived in an old compact trace, never invent q/v."""
    index = request["source_trace_index"]
    if index is None:
        return None
    saved = trace[index]
    actual_commands = np.asarray(request["commands"], dtype=np.float64)
    saved_commands = np.asarray(saved["commands"], dtype=np.float64)
    require(
        actual_commands.shape == saved_commands.shape and actual_commands.tobytes() == saved_commands.tobytes(),
        "Endpoint commands must retain exact saved bytes",
    )
    differences = {}
    for key in ("tcp_errors", "safety_min"):
        actual, expected = np.asarray(classification[key]), np.asarray(saved[key])
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        differences[key] = float(np.max(np.abs(actual - expected)))
    for key in ("prefix_length", "joint_feasible", "feasible"):
        require(classification[key] == saved[key], f"Strict saved endpoint label: {key}")
    return dict(
        source_trace_index=index,
        fields_checked=["commands", "tcp_errors", "safety_min", "prefix_length", "joint_feasible", "feasible"],
        commands_exact_bytes=True,
        tcp_errors_max_abs_difference=differences["tcp_errors"],
        safety_min_abs_difference=differences["safety_min"],
        strict_labels_equal=True,
        old_full_rollout_available=False,
    )


def validate_batch(batch, diagnostic, producers):
    """Validate the fixed real-data counts before any new numerical response."""
    old = {
        np.asarray(row["commands"], dtype=np.float64).tobytes()
        for source in producers.values()
        for row in source["answer"]["evaluated_candidates"]
    }
    keys = [np.asarray(request["commands"], dtype=np.float64).tobytes() for request in batch]
    require(len(batch) == len(set(keys)) == 96, "All96 fixed requests must remain distinct")
    require(len(set(keys) & old) == 32 and len(set(keys) - old) == 64, "32 old endpoints and64 new interior commands")
    for name, counts in EXPECTED_COUNTS.items():
        entries = [request for request in batch if request["case_name"] == name]
        horizon, target = (12, 9) if name == "prefix" else (13, 10)
        require(len(entries) == counts["response_evaluations"], "All registered directions retained")
        require(
            all(e["horizon"] == horizon and e["target_index"] == target for e in entries), "Original cursor9 target"
        )
        for entry in entries:
            pair = diagnostic["cases"][name]["pairs"][entry["pair_index"]]
            base = producers[name]["answer"]["evaluated_candidates"][entry["base_index"]]
            require(
                pair["main_subset"] is True
                and pair["phase_target_cursor"] == entry["prefix_before"] == target
                and base["joint_feasible"] is True
                and base["safety_min"] >= 1e-8
                and base["prefix_length"] >= entry["prefix_before"],
                "Only preregistered feasible-base main pairs",
            )
            key = np.asarray(entry["commands"], dtype=np.float64).tobytes()
            require((key in old) == (entry["source_trace_index"] is not None), "Only endpoints were saved previously")
    return dict(response_evaluations=96, new_unique_commands=64, repeated_saved_endpoints=32)


def evaluate_case(
    requests,
    source,
    interface,
    dynamics,
    *,
    rollout_fn=rollout_native_goal_trajectory,
    score_fn=score_backtracking_response,
):
    """Charge and reset every request, including repeated old endpoints and failures."""
    q = np.asarray(source["current_q"], dtype=float).copy()
    velocity = np.asarray(source["current_velocity"], dtype=float).copy()
    latch = np.asarray(source["latched_rotation_goal"], dtype=float).copy()
    limits, torque = np.asarray(source["joint_limits"]), np.asarray(source["torque_limits"])
    start_queries = dynamics.queries
    current = dynamics(q.copy(), velocity.copy())
    require(dynamics.queries - start_queries == 1, "Exactly one initial current query per case")
    for key, value in current.items():
        np.testing.assert_allclose(value, source["current_quantities"][key], rtol=0, atol=1e-10)
    result = dict(
        current_quantities=current, entries=[], response_evaluations=0, numerical_inner_steps=0, query_count=1
    )
    for request in requests:
        commands = np.asarray(request["commands"], dtype=float).copy()
        before = dynamics.queries
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
        horizon = request["horizon"]
        require(
            dynamics.queries - before == 26 * horizon,
            "Every response charges all25 inner and one endpoint query per period",
        )
        classification = score_fn(
            rollout,
            commands,
            np.asarray(source["previous_command"])[:6].copy(),
            interface,
            limits[:, 0].copy(),
            limits[:, 1].copy(),
            past_inner_error_rad=np.asarray(source["past_inner_error_rad"]).copy(),
            terminal_normals=np.asarray(source["terminal_normals"]).copy(),
            desired_tcp=np.asarray(source["desired_tcp"]).copy(),
            site_from_tcp=np.asarray(source["site_from_tcp"]).copy(),
            prefix_before=request["prefix_before"],
            target_index=request["target_index"],
            base_target_max=request["base_target_max"],
        )
        dual = None
        if source.get("case_contract") is not None:
            dual = score_phase_response(rollout["pose"], source["case_contract"], source["site_from_tcp"])
        endpoint = compare_saved_endpoint(request, classification, source["trace"])
        result["entries"].append(
            dict(
                request=request,
                rollout=rollout,
                classification=classification,
                dual_metrics=dual,
                endpoint_match=endpoint,
            )
        )
        result["response_evaluations"] += 1
        result["numerical_inner_steps"] += 25 * horizon
        result["query_count"] = dynamics.queries - start_queries
    require(
        result["query_count"] == 1 + sum(26 * r["horizon"] for r in requests),
        "No cache, extra queries or skipped responses",
    )
    return result


def main(dry=False):
    strict_native_environment()
    require(
        all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()),
        "Explicit registered single-thread CPU environment",
    )
    diagnostic, producers, registrations, refs = load_prerequisites()
    batch = build_batch(diagnostic, producers)
    batch_counts = validate_batch(batch, diagnostic, producers)
    finish(refs)
    if dry:
        print(json.dumps(dict(dry_run=True, passed=True, source_count=len(refs), **batch_counts)), flush=True)
        return
    require(not ROOT.exists(), "No overwrite or automatic retry")
    registration = publish_result(
        ROOT / "registration.json",
        dict(
            diagnostic_ref=DIAGNOSTIC_REF,
            plan_ref=reference(PLAN),
            producer_refs=read(diagnostic["registration_ref"])["producer_refs"],
            source_refs=[reference(REPO / relative) for relative in SOURCES],
            input_refs=list(refs.values()),
            seed=42,
            expected_counts=EXPECTED_COUNTS,
            alphas=[0, 0.0625, 0.125, 0.25, 0.5, 1],
            endpoint=368,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        cases={},
        response_evaluations=0,
        numerical_inner_steps=0,
        query_count=0,
        new_unique_commands=64,
        repeated_saved_endpoints=32,
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
        for name in CASE_ORDER:
            original, source_reg = producers[name], registrations[name]
            control = read(source_reg["control_ref"])["contract"]
            interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
            dynamics = MujocoRobotDynamics(
                checked(source_reg["xml_ref"]),
                control["controller_site"],
                [f"robot0_joint{k}" for k in range(1, 8)],
                original["fixed_gripper_state"],
            )
            require(dynamics.model.opt.timestep == 0.002, "Original2ms XML")
            with np.load(checked(source_reg["states_ref"]), allow_pickle=False) as archive:
                state = archive["sim_states"][368].copy()
            np.testing.assert_array_equal(original["current_q"], state[1 + dynamics.qids])
            np.testing.assert_array_equal(original["current_velocity"], state[1 + dynamics.model.nq + dynamics.dofs])
            for joint_name, fixed in original["fixed_gripper_state"].items():
                jid = dynamics.mj.mj_name2id(dynamics.model, dynamics.mj.mjtObj.mjOBJ_JOINT, joint_name)
                require(jid >= 0, "Original actual368 finger coordinate")
                np.testing.assert_array_equal(
                    fixed,
                    [
                        state[1 + dynamics.model.jnt_qposadr[jid]],
                        state[1 + dynamics.model.nq + dynamics.model.jnt_dofadr[jid]],
                    ],
                )
            np.testing.assert_array_equal(original["joint_limits"], dynamics.model.jnt_range[dynamics.joints])
            source = dict(
                original,
                trace=original["answer"]["evaluated_candidates"],
                desired_tcp=original["desired_tcp"] if name == "prefix" else original["case_contract"]["desired_tcp"],
            )
            result["cases"][name] = evaluate_case(
                [entry for entry in batch if entry["case_name"] == name], source, interface, dynamics
            )
            for key, expected in EXPECTED_COUNTS[name].items():
                require(result["cases"][name][key] == expected, f"Exact registered {name} {key}")
                result[key] += expected
            print(
                json.dumps(dict(case_name=name, **EXPECTED_COUNTS[name], elapsed_s=time.monotonic() - started)),
                flush=True,
            )
        require(
            result["response_evaluations"] == 96
            and result["numerical_inner_steps"] == 29700
            and result["query_count"] == 30891,
            "Full fixed batch",
        )
        strict_native_environment()
        finish(refs)
        result.update(
            complete=True, passed=True, source_refs_unchanged=list(refs.values()), elapsed_s=time.monotonic() - started
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(ROOT / "verified.json", result)
    print(json.dumps(dict(complete=True, passed=True, response_evaluations=96, control_qualified=False)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    main(parser.parse_args().dry)
