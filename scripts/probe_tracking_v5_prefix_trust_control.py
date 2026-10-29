"""Run one registered H12 comparison arm from the same verified saved warm."""

import argparse
import json
import os
import time

import numpy as np

from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface
from s2a.prefix_preserving_trajectory_plan import solve_prefix_preserving_trajectory_response
from s2a.prefix_trust_control_evidence import (
    BACKTRACKING_REF,
    CPU_ENVIRONMENT,
    PLAN,
    REPO,
    SOURCES,
    case_root,
    checked,
    finish,
    load_prerequisites,
    read,
    reference,
    require,
)
from s2a.preview_result_publication import publish_result
from s2a.terminal_development_execution import CONTROL_REF


def admit_warm(source, warm, interface):
    warm = np.asarray(warm, dtype=float)
    require(warm.shape == (12, 6) and np.all(np.isfinite(warm)), "Same complete finite H12 warm")
    previous = np.asarray(source["previous_command"], dtype=float)[:6]
    lo, hi = np.tile(interface.input_min, (12, 1)), np.tile(interface.input_max, (12, 1))
    lo[0], hi[0] = np.maximum(lo[0], previous - 0.25), np.minimum(hi[0], previous + 0.25)
    require(
        np.clip(warm, lo, hi).tobytes() == warm.tobytes(), "Frozen solver start clipping must preserve exact warm bytes"
    )
    require(
        np.all(np.abs(np.diff(np.vstack((previous, warm)), axis=0)) <= 0.25 + 1e-12), "Original warm rate admission"
    )
    return warm.copy()


def run_case(
    case_name,
    source,
    warm,
    interface,
    dynamics,
    *,
    solver_fn=None,
    rollout_fn=rollout_native_goal_trajectory,
    progress=None,
):
    """Reset and charge every actual predictor invocation; the solver owns its cache."""
    require(case_name in ("baseline", "candidate"), "One registered comparison arm")
    warm = admit_warm(source, warm, interface)
    if solver_fn is None:
        if case_name == "baseline":
            solver_fn = solve_prefix_preserving_trajectory_response
        else:
            from s2a.prefix_trust_region_plan import solve_prefix_trust_region_response

            solver_fn = solve_prefix_trust_region_response
    q, velocity = np.asarray(source["current_q"]).copy(), np.asarray(source["current_velocity"]).copy()
    latch = np.asarray(source["latched_rotation_goal"]).copy()
    limits, torque = np.asarray(source["joint_limits"]), np.asarray(source["torque_limits"])
    start_queries = dynamics.queries
    result = {} if progress is None else progress
    result.update(
        prediction_calls=0, response_evaluations=0, total_iterations=0, numerical_inner_steps=0, query_count=0
    )
    current = dynamics(q.copy(), velocity.copy())
    result["query_count"] = dynamics.queries - start_queries
    require(result["query_count"] == 1, "Exactly one current query per arm")
    for key, value in current.items():
        np.testing.assert_allclose(value, source["current_quantities"][key], atol=1e-10, rtol=0)
    result["current_quantities"] = current
    evaluated, seen = [], set()
    started = time.monotonic()

    def predict(commands):
        commands = np.asarray(commands, dtype=float).copy()
        result["prediction_calls"] += 1
        require(commands.shape == (12, 6) and np.all(np.isfinite(commands)), "Complete finite H12 predictor request")
        key = commands.tobytes()
        require(key not in seen, "Solver cache must prevent duplicate raw responses; no hidden outer cache")
        require(len(evaluated) < 4000, "All actual responses share the fixed4000 budget")
        if not evaluated:
            require(key == warm.tobytes(), "First billed response must use the exact common warm")
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
        require(dynamics.queries - before == 312, "Every full H12 response bills300 inner plus12 endpoint queries")
        seen.add(key)
        evaluated.append(key)
        if len(evaluated) == 1 or len(evaluated) % 100 == 0:
            print(
                json.dumps(dict(case_name=case_name, responses=len(evaluated), elapsed_s=time.monotonic() - started)),
                flush=True,
            )
        return rollout

    answer = solver_fn(
        predict,
        np.asarray(source["previous_command"])[:6].copy(),
        warm.copy(),
        interface,
        limits[:, 0].copy(),
        limits[:, 1].copy(),
        np.asarray(source["desired_tcp"]).copy(),
        np.asarray(source["site_from_tcp"]).copy(),
        past_inner_error_rad=np.asarray(source["past_inner_error_rad"]).copy(),
        terminal_normals=np.asarray(source["terminal_normals"]).copy(),
        max_evaluations=4000,
        max_iterations=60,
        finite_difference_step=1e-4,
    )
    result["answer"] = answer
    result["total_iterations"] = answer["total_iterations"]
    trace = answer["evaluated_candidates"]
    require(
        1
        <= result["response_evaluations"]
        == result["prediction_calls"]
        == answer["response_evaluations"]
        == len(trace)
        <= 4000
        and result["numerical_inner_steps"] == 300 * len(trace)
        and result["query_count"] == 1 + 312 * len(trace)
        and 0 <= result["total_iterations"] <= 60,
        "All actual predictor calls, solver trace and finite budgets agree",
    )
    require(
        evaluated == [np.asarray(row["commands"], dtype=float).tobytes() for row in trace],
        "Every actual response retained once in its original order",
    )
    return result


def main(case_name, dry=False):
    strict_native_environment()
    require(
        all(os.environ.get(key) == value for key, value in CPU_ENVIRONMENT.items()),
        "Registered single-thread CPU environment",
    )
    source, source_reg, backtracking, refs = load_prerequisites()
    root = case_root(case_name)
    require(not root.exists(), "No overwrite or automatic retry")
    control = read(source_reg["control_ref"])["contract"]
    interface = OSCInterface(*(control[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    warm = admit_warm(source, backtracking["cases"]["prefix"]["entries"][55]["request"]["commands"], interface)
    finish(refs)
    if dry:
        print(
            json.dumps(dict(dry_run=True, passed=True, case_name=case_name, source_count=len(refs), responses=0)),
            flush=True,
        )
        return
    registration = publish_result(
        root / "registration.json",
        dict(
            case_name=case_name,
            backtracking_ref=BACKTRACKING_REF,
            plan_ref=reference(PLAN),
            seed=42,
            original_prefix_registration_ref=source["registration_ref"],
            source_refs=[reference(REPO / relative) for relative in SOURCES],
            input_refs=list(refs.values()),
            warm_locator="cases.prefix.entries[55].request.commands",
            endpoint=368,
            horizon=12,
            response_budget=4000,
            iteration_budget=60,
            stage_budget=60,
            actual_consumed_reference_count=0,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        case_name=case_name,
        registration_ref=registration,
        warm=warm,
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
            checked(source_reg["xml_ref"]),
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
            require(jid >= 0, "Original actual368 finger coordinate")
            np.testing.assert_array_equal(
                fixed,
                [
                    state[1 + dynamics.model.jnt_qposadr[jid]],
                    state[1 + dynamics.model.nq + dynamics.model.jnt_dofadr[jid]],
                ],
            )
        run_case(case_name, source, warm, interface, dynamics, progress=result)
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
            dict(complete=True, passed=True, case_name=case_name, witness_found=result["answer"]["witness_found"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args.case, args.dry)
