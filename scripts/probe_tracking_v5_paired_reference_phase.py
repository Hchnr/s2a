"""One registered H13 reference-phase allocation, with a single audited warm."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.geometry import inverse
from s2a.mujoco_robot_dynamics import MujocoRobotDynamics
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.native_goal_joint_response import native_goal_rotation
from s2a.native_goal_trajectory import rollout_native_goal_trajectory
from s2a.osc import OSCInterface
from s2a.paired_reference_phase import (
    assess_phase_warm,
    prepare_paired_reference_phases,
    score_phase_response,
)
from s2a.prefix_preserving_trajectory_plan import solve_prefix_preserving_trajectory_response
from s2a.preview_result_publication import publish_result as publish
from s2a.terminal_development_execution import CONTROL_REF

PREFIX_OWNER = REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_check_execution_20261028/result.json"
PREFIX_RESULT = REPO / "outputs/tracking_v5_prefix_preserving_trajectory_control_20261028/verified.json"
PHASE_RESULT = REPO / "outputs/tracking_v5_saved_trajectory_phase_20261028/result.json"
PLAN = REPO / "plans/v0.0.1/paired_reference_phase_20261028.md"


class CachedTrajectoryResponses:
    """Charge each exact H13 command once, including the preflight warm.

    The frozen solver has its own cache. This outer cache bridges the one
    preflight call and its identical solver-initial call without spending a
    second response. It does not merge near commands or signed-zero bytes.
    """

    def __init__(self, predict, *, max_evaluations=4000):
        require(type(max_evaluations) is int and max_evaluations > 0, "Positive unique-response budget")
        self.predict = predict
        self.max_evaluations = max_evaluations
        self.calls = 0
        self.hits = 0
        self.responses = {}

    @staticmethod
    def key(commands):
        commands = np.asarray(commands, dtype=float)
        require(commands.shape == (13, 6) and np.all(np.isfinite(commands)), "Finite thirteen-command trajectory")
        return commands.tobytes()

    def __call__(self, commands):
        key = self.key(commands)
        self.calls += 1
        if key in self.responses:
            self.hits += 1
            return self.responses[key]
        require(len(self.responses) < self.max_evaluations, "All unique responses share the fixed budget")
        response = self.predict(np.asarray(commands, dtype=float).copy())
        self.responses[key] = response
        return response


def closed_owner(ref, remember=checked):
    """Require the exact successful CPU child and its actual wait/reap receipts."""
    remember(ref)
    owner = read(ref)
    require(owner["complete"] is owner["passed"] is True, "Prerequisite owner fully closed")
    refs = [owner["registration_ref"], owner["result_ref"], *owner["owned_command"].values()]
    for item in refs:
        remember(item)
    launch = read(owner["owned_command"]["launch"])
    exited = read(owner["owned_command"]["exit"])
    release = read(owner["owned_command"]["release"])
    require(
        exited["actual_wait"] is True
        and exited["returncode"] == 0
        and exited["automatic_retry"] is False
        and launch["automatic_retry"] is False
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["pid"] == launch["child"]["pid"]
        and release["start_ticks"] == launch["child"]["start_ticks"]
        and launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] == ""
        and launch["selected_environment"]["LD_LIBRARY_PATH"] == "",
        "Same CPU prerequisite actual wait0/reap0, without signal or retry",
    )
    result = read(owner["result_ref"])
    require(result["complete"] is result["passed"] is True, "Complete prerequisite numerical receipt")
    for path, digest in read(owner["registration_ref"])["source_sha256"].items():
        remember(dict(path=path, sha256=digest))
    return owner, result


def phase_prerequisite(prefix_ref, trajectory_ref, remember=checked):
    """Bind the completed geometry-only diagnosis, without inventing an owner."""
    ref = reference(PHASE_RESULT)
    remember(ref)
    phase = read(ref)
    remember(phase["registration_ref"])
    require(
        phase["complete"] is phase["passed"] is phase["source_unchanged"] is phase["inputs_unchanged"] is True
        and phase["control_qualified"] is phase["retiming_qualified"] is False
        and all(
            phase[key] == 0
            for key in ("physics_steps", "world_model_calls", "fresh_cases", "new_commands", "new_forward_responses")
        )
        and phase["input_refs_before"] == phase["input_refs_after"]
        and phase["source_refs_before"] == phase["source_refs_after"]
        and prefix_ref in phase["input_refs_before"]
        and trajectory_ref in phase["input_refs_before"],
        "Closed geometry diagnosis binds these two saved trajectories and adds no responses",
    )
    for item in [*phase["input_refs_before"], *phase["source_refs_before"]]:
        remember(item)
    return ref


def main(case_name):
    strict_native_environment()
    require(case_name in ("early", "late"), "One registered case")
    root = REPO / f"outputs/tracking_v5_paired_reference_phase_{case_name}_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    tracked = {}

    def remember(ref):
        checked(ref)
        require(ref["path"] not in tracked or tracked[ref["path"]] == ref, "One immutable hash per source")
        tracked[ref["path"]] = ref
        return ref

    prefix_owner_ref = remember(reference(PREFIX_OWNER))
    prefix_owner, prefix_check = closed_owner(prefix_owner_ref, remember)
    prefix_check_reg = read(remember(prefix_check["registration_ref"]))
    prefix_ref = remember(reference(PREFIX_RESULT))
    require(
        prefix_check_reg["producer_ref"] == prefix_ref
        and prefix_check["trace_count"] == 4000
        and prefix_check["numerical_inner_steps"] == 1200000
        and prefix_check["mj_forward_queries"] == 1248001
        and prefix_check["witness_found"] is prefix_check["control_qualified"] is False,
        "Full independently closed prefix failure before paired allocation",
    )
    for ref in prefix_check["source_refs_unchanged"]:
        remember(ref)
    producer_owner, prefix = closed_owner(prefix_check_reg["producer_owner_ref"], remember)
    require(producer_owner["result_ref"] == prefix_ref, "Exact closed prefix producer")
    prefix_reg = read(remember(prefix["registration_ref"]))
    trajectory_ref = remember(prefix_reg["prior_trajectory_ref"])
    trajectory = read(trajectory_ref)
    phase_ref = phase_prerequisite(prefix_ref, trajectory_ref, remember)
    causal_keys = (
        "prior_trajectory_check_owner_ref",
        "prior_trajectory_check_ref",
        "prior_trajectory_ref",
        "early_numeric_ref",
        "early_independent_owner_ref",
        "physical_independent_owner_ref",
        "original_producer_ref",
        "manifest_ref",
        "states_ref",
        "original_rows_ref",
        "initial_ref",
        "xml_ref",
        "control_ref",
        "tool_ref",
        "chunks_ref",
    )
    for key in causal_keys:
        remember(prefix_reg[key])
    for ref in [*prefix_reg["native_source_refs"], prefix_reg["native_transform_ref"]]:
        remember(ref)
    common_keys = (
        "current_q",
        "current_velocity",
        "fixed_gripper_state",
        "current_quantities",
        "initial_joint",
        "kp",
        "kd",
        "torque_limits",
        "joint_limits",
        "site_from_tcp",
        "previous_command",
        "previous_site",
        "previous_physical_delta",
        "latched_rotation_goal",
        "past_inner_error_rad",
        "terminal_normals",
    )
    for key in common_keys:
        require(prefix[key] == trajectory[key], f"Same original H12 causal input: {key}")
    warm = np.asarray(prefix["warm"], dtype=float)
    np.testing.assert_array_equal(warm, trajectory["warm"])
    early = read(prefix_reg["early_numeric_ref"])
    entry = next(e for e in early["entries"] if e["step"] == 368 and e["horizon"] == 12)
    np.testing.assert_array_equal(warm, entry["answer"]["planned_commands"])
    profile = np.zeros((25, 7))
    normals = np.eye(7)[6:]
    np.testing.assert_array_equal(prefix["past_inner_error_rad"], profile)
    np.testing.assert_array_equal(prefix["terminal_normals"], normals)
    with checked(prefix_reg["original_rows_ref"]).open() as stream:
        for index, line in enumerate(stream):
            if index == 367:
                previous_row = json.loads(line)
                break
        else:
            raise ValueError("Missing completed actual367 row")
    previous = np.asarray(prefix["previous_command"])
    latch = np.asarray(prefix["latched_rotation_goal"])
    np.testing.assert_array_equal(previous, previous_row["command"])
    np.testing.assert_array_equal(latch, previous_row["controller_goal_after"])
    np.testing.assert_array_equal(prefix["previous_site"], previous_row["qualified_forward_input"]["x_site_pose"][0])
    contract_record = read(prefix_reg["control_ref"])
    control = contract_record["contract"]
    interface = OSCInterface(*(control[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    np.testing.assert_array_equal(interface.decode(previous[:6]), prefix["previous_physical_delta"])
    np.testing.assert_allclose(
        native_goal_rotation(interface.decode(previous[:6])[3:]) @ np.asarray(prefix["previous_site"])[:3, :3],
        latch,
        atol=1e-10,
        rtol=0,
    )
    tool = inverse(np.asarray(contract_record["source_from_site"])) @ np.asarray(
        read(prefix_reg["tool_ref"])["source_from_tcp"]
    )
    np.testing.assert_array_equal(tool, prefix["site_from_tcp"])
    chunk = read(prefix_reg["chunks_ref"])[46]
    require(chunk["step"] == 368, "Already generated chunk46 at actual368")
    targets, openings = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["openings"])
    np.testing.assert_array_equal(targets[:12], prefix["desired_tcp"])
    np.testing.assert_array_equal(openings[:13], np.ones(13))
    source_refs = [remember(reference(Path(__file__))), remember(reference(PLAN))]
    for relative in (
        "src/s2a/paired_reference_phase.py",
        "src/s2a/native_goal_trajectory.py",
        "src/s2a/prefix_preserving_trajectory_plan.py",
        "src/s2a/native_goal_joint_response.py",
        "src/s2a/nonlinear_joint_response.py",
        "src/s2a/mujoco_robot_dynamics.py",
        "src/s2a/joint_preview_constraint_audit.py",
    ):
        source_refs.append(remember(reference(REPO / relative)))
    registration = publish(
        root / "registration.json",
        dict(
            case_name=case_name,
            prior_prefix_check_owner_ref=prefix_owner_ref,
            prior_prefix_check_ref=prefix_owner["result_ref"],
            prior_prefix_ref=prefix_ref,
            saved_phase_diagnostic_ref=phase_ref,
            **{key: prefix_reg[key] for key in causal_keys},
            native_source_refs=prefix_reg["native_source_refs"],
            native_transform_ref=prefix_reg["native_transform_ref"],
            source_refs=source_refs,
            input_refs=list(tracked.values()),
            seed=42,
            endpoint=368,
            cursor=0,
            horizon=13,
            initial_history_completed_periods=0,
            history_policy="Actual368 and completed367 only; zero error profile; no later actual state or command",
            warm_policy="Original H12 warm only, with one registered adjacent mean; exact-byte preflight before any query",
            response_policy="Carry predicted q/v/native goals for13 periods of25x2ms; fixed actual368 finger q/v",
            gripper_policy="Opening uses the same phase; targets0..12 are1; archive actual367 motor state; fixed fingers are an approximation",
            search=dict(
                response_budget=4000,
                total_iterations=60,
                difference_step=1e-4,
                optimization_safety_buffer=1e-8,
                ftol=1e-12,
                starts=1,
                restoration=False,
            ),
            constraints=dict(
                rate=0.25,
                joint_margin=0.05,
                braking_time=0.1,
                position_cap=0.01,
                rotation_cap=0.05,
                pose_caps_horizon=13,
                terminal_normals=normals,
            ),
            actual_consumed_reference_count=0,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration,
        case_name=case_name,
        **{key: prefix[key] for key in common_keys},
        source_warm12=warm,
        source_targets_tcp=targets,
        source_openings=openings,
        actual367_motor_target_after=previous_row["motor_target_after"],
        actual367_motor_opening_after=previous_row["motor_opening_after"],
        actual367_gripper_command=float(previous[6]),
        history_before_search=dict(
            completed_periods=0, last_completed_step=None, inner_position_error_rad=profile, future_guarantee=False
        ),
        case_contract=None,
        command_admission=None,
        warm_preflight=None,
        warm_admission=None,
        answer=None,
        trace=[],
        dual_metrics={},
        phases=[],
        response_evaluations=0,
        total_iterations=0,
        numerical_inner_steps=0,
        prediction_calls=0,
        cache_hits=0,
        query_count=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        actual_consumed_reference_count=0,
        witness_found=False,
        control_qualified=False,
        retimed_control_qualified=False,
        original_timed_control_qualified=False,
    )
    started = time.monotonic()
    try:
        try:
            case = prepare_paired_reference_phases(
                targets,
                openings,
                warm,
                previous=previous[:6],
                interface=interface,
                case_name=case_name,
            )[case_name]
        except ValueError as error:
            result.update(status="command_admission_failed", command_admission=dict(admitted=False, reason=str(error)))
        else:
            result.update(case_contract=case, command_admission=dict(admitted=True, reason=None))
            guard = read(CONTROL_REF)["cv2_guard_registration"]
            result["environment_admission"] = restore_cv2_environment(
                guard["exact_probe_ref"], guard["runtime_source_refs"]
            )
            os.environ.pop("MUJOCO_GL", None)
            os.environ.pop("PYOPENGL_PLATFORM", None)
            query = MujocoRobotDynamics(
                checked(prefix_reg["xml_ref"]),
                control["controller_site"],
                [f"robot0_joint{k}" for k in range(1, 8)],
                prefix["fixed_gripper_state"],
            )
            require(query.model.opt.timestep == 0.002, "Original2ms XML")
            q, velocity = np.asarray(prefix["current_q"]), np.asarray(prefix["current_velocity"])
            with np.load(checked(prefix_reg["states_ref"]), allow_pickle=False) as archive:
                state = archive["sim_states"][368].copy()
            np.testing.assert_array_equal(q, state[1 + query.qids])
            np.testing.assert_array_equal(velocity, state[1 + query.model.nq + query.dofs])
            for name, fixed in prefix["fixed_gripper_state"].items():
                jid = query.mj.mj_name2id(query.model, query.mj.mjtObj.mjOBJ_JOINT, name)
                require(jid >= 0, "Original current finger coordinate")
                np.testing.assert_array_equal(
                    fixed,
                    [
                        state[1 + query.model.jnt_qposadr[jid]],
                        state[1 + query.model.nq + query.model.jnt_dofadr[jid]],
                    ],
                )
            current = query(q, velocity)
            result["query_count"] = query.queries
            for key, value in current.items():
                np.testing.assert_allclose(value, prefix["current_quantities"][key], rtol=0, atol=1e-10)
            limits = query.model.jnt_range[query.joints]
            np.testing.assert_array_equal(limits, prefix["joint_limits"])
            result["current_quantities"] = current
            torque = np.asarray(prefix["torque_limits"])

            def raw_predict(commands):
                return rollout_native_goal_trajectory(
                    q,
                    velocity,
                    commands,
                    interface,
                    np.asarray(prefix["initial_joint"]),
                    np.asarray(prefix["kp"]),
                    np.asarray(prefix["kd"]),
                    query,
                    torque[:, 0],
                    torque[:, 1],
                    latched_rotation_goal=latch,
                )

            cached_predict = CachedTrajectoryResponses(raw_predict)
            cache = cached_predict.responses

            def predict(commands):
                before = len(cache)
                response = cached_predict(commands)
                result["prediction_calls"] = cached_predict.calls
                result["cache_hits"] = cached_predict.hits
                result["response_evaluations"] = len(cache)
                result["numerical_inner_steps"] = len(cache) * 325
                result["query_count"] = query.queries
                if len(cache) != before and (len(cache) == 1 or len(cache) % 100 == 0):
                    print(
                        json.dumps(
                            dict(
                                case_name=case_name,
                                responses_completed=len(cache),
                                elapsed_s=time.monotonic() - started,
                            )
                        ),
                        flush=True,
                    )
                return response

            preflight = assess_phase_warm(
                predict(case["warm"]),
                case,
                previous[:6],
                interface,
                limits[:, 0],
                limits[:, 1],
                tool,
                past_inner_error_rad=profile,
                terminal_normals=normals,
            )
            result["warm_preflight"] = preflight["initial"]
            result["warm_admission"] = {
                key: preflight[key]
                for key in (
                    "joint_feasible",
                    "safety_min",
                    "complete_witness",
                    "admitted",
                    "status",
                    "rejection_reason",
                )
            }
            result["dual_metrics"]["warm"] = preflight["metrics"]
            if not preflight["admitted"]:
                result["status"] = "warm_admission_failed"
                item = preflight["initial"]
                result["trace"] = [
                    dict(
                        commands=item["commands"],
                        safety_min=preflight["safety_min"],
                        joint_feasible=preflight["joint_feasible"],
                        tcp_errors=item["tcp_errors"],
                        prefix_length=item["prefix_length"],
                        feasible=item["feasible"],
                        dual_metrics=preflight["metrics"],
                    )
                ]
                require(result["prediction_calls"] == 1 and result["cache_hits"] == 0, "One rejected warm response")
            else:
                answer = solve_prefix_preserving_trajectory_response(
                    predict,
                    previous[:6],
                    case["warm"],
                    interface,
                    limits[:, 0],
                    limits[:, 1],
                    case["desired_tcp"],
                    tool,
                    past_inner_error_rad=profile,
                    terminal_normals=normals,
                    max_evaluations=4000,
                    max_iterations=60,
                    finite_difference_step=1e-4,
                )
                result.update(
                    answer=answer,
                    status=answer["status"],
                    witness_found=answer["witness_found"],
                    total_iterations=answer["total_iterations"],
                    phases=answer["phases"],
                )
                require(
                    answer["response_evaluations"] == len(cache), "Preflight warm reused by the solver, charged once"
                )
                require(
                    result["prediction_calls"] == len(cache) + 1 and result["cache_hits"] == 1,
                    "Exactly one preflight cache reuse",
                )
                require(
                    np.asarray(answer["used_start"]).tobytes() == np.asarray(case["warm"]).tobytes(),
                    "No solver roundoff clipping of the warm",
                )
                for item in answer["evaluated_candidates"]:
                    rollout = cache[np.asarray(item["commands"], dtype=float).tobytes()]
                    result["trace"].append(dict(item, dual_metrics=score_phase_response(rollout["pose"], case, tool)))
                for name in ("initial", "selected", "closest_safety_attempt"):
                    result["dual_metrics"][name] = score_phase_response(answer[name]["rollout"]["pose"], case, tool)
            require(
                result["response_evaluations"] <= 4000 and result["total_iterations"] <= 60, "Fixed per-case budget"
            )
            require(
                query.queries == 1 + 338 * len(cache), "One current query and26 native queries per predicted period"
            )
        result["search_seconds"] = time.monotonic() - started
        for ref in tracked.values():
            checked(ref)
        result.update(complete=True, passed=True, source_refs_unchanged=list(tracked.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "verified.json", result)
    print(
        json.dumps(
            dict(
                complete=True,
                passed=True,
                case_name=case_name,
                status=result["status"],
                response_evaluations=result["response_evaluations"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("early", "late"))
    main(parser.parse_args().case)
