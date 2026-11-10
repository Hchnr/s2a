"""Compare matched old/new causal references and replay the executed actions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import inverse, pose_error
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import (
    current_joint_site_state,
    current_robot_dynamics,
    run_continuation,
)
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_causal_event_feedback as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
QA = REPO / "outputs/qa/causal_event_feedback_20261110/result.json"
DRY = REPO / "outputs/causal_event_feedback_dry_20261110/result.json"
ROOT = REPO / "outputs/causal_event_feedback_20261110"
BRANCHES = ((0, "latest"), (0, "remembered"), (5, "latest"), (5, "remembered"))
FORBIDDEN_CHECKER_MODULES = (
    "s2a.research.joint_control.preview.refined_feedback",
    "s2a.research.joint_control.preview.quality_feedback_refiner",
    "s2a.research.joint_control.preview.loaded_gripper_feedback",
    "s2a.research.joint_control.tracking.loaded_velocity_control",
)


def source_refs():
    return binding.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            REPO / "src/s2a/research/joint_control/preview/loaded_gripper_feedback.py",
            REPO / "src/s2a/research/joint_control/preview/refined_feedback.py",
            REPO / "src/s2a/research/joint_control/preview/quality_feedback_refiner.py",
            REPO / "src/s2a/research/joint_control/preview/completed_history.py",
            REPO / "src/s2a/research/gripper/loaded_gripper_transaction.py",
            REPO / "src/s2a/research/gripper/causal_loaded_history.py",
            REPO / "scripts/simulation/runtime/reconstruct_causal_gripper_history.py",
            REPO / "src/s2a/research/native_runtime/development_continuation.py",
            REPO / "src/s2a/research/native_runtime/final_prefix_inner.py",
            REPO / "tests/research/joint_control/preview/test_loaded_gripper_feedback.py",
            REPO / "tests/research/joint_control/preview/test_refined_feedback.py",
            REPO / "tests/research/gripper/test_loaded_gripper_transaction.py",
            REPO / "tests/research/native_runtime/test_causal_event_feedback_runtime.py",
            REPO / "tests/research/native_runtime/test_final_prefix_inner.py",
        )
    ]


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "cases": [case["spec"] for case in inputs["cases"]],
        "source_sha256": inputs["source_sha256"],
        "asset_refs": inputs["asset_refs"],
    }


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One registered case and reference branch")
    return f"case_{index:02d}_{mode}"


def measured_gripper(env):
    """Read physical values for scoring without importing either policy."""
    robot, data = env.robots[0], env.sim.data
    q = np.asarray(data.qpos)[robot._ref_gripper_joint_pos_indexes].copy()
    v = np.asarray(data.qvel)[robot._ref_gripper_joint_vel_indexes].copy()
    motor = np.asarray(robot.gripper.current_action, dtype=float).copy()
    prefix.require(
        q.shape == v.shape == motor.shape == (2,) and np.isfinite([*q, *v, *motor]).all(), "Finite gripper state"
    )
    return {"opening": float(np.clip((q[0] - q[1] - 0.001) / 0.079, 0, 1)), "qpos": q, "qvel": v, "motor_target": motor}


def score_completed(env, case, selected, mode, offset, own_error, *, expected=None):
    """Score both already generated references after, never before, dispatch."""
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, _velocity, site = current_robot_dynamics(env, jacobian)
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    actual_tcp = site @ site_from_tcp
    errors, opening_targets = {}, {}
    for name in ("latest", "remembered"):
        ref = case["references"][name]
        index = ref["cursor"] + offset
        errors[name] = pose_error(ref["poses"][index], actual_tcp)
        opening_targets[name] = float(ref["openings"][index])
    np.testing.assert_allclose(errors[mode], own_error, atol=1e-10, rtol=0)
    gripper = measured_gripper(env)
    result = {
        "actual_tcp": actual_tcp,
        "reference_errors": errors,
        "opening_targets": opening_targets,
        "opening_errors": {name: gripper["opening"] - value for name, value in opening_targets.items()},
        "gripper": gripper,
        "selected_absolute_endpoint": selected["spec"]["prefix_steps"] + offset + 1,
    }
    if expected is not None:
        np.testing.assert_allclose(result["actual_tcp"], expected["actual_tcp"], atol=1e-10, rtol=0)
        for name in ("latest", "remembered"):
            np.testing.assert_allclose(errors[name], expected["reference_errors"][name], atol=1e-10, rtol=0)
            np.testing.assert_allclose(opening_targets[name], expected["opening_targets"][name], atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                result["opening_errors"][name], expected["opening_errors"][name], atol=1e-10, rtol=0
            )
        for key in ("opening", "qpos", "qvel", "motor_target"):
            np.testing.assert_allclose(gripper[key], expected["gripper"][key], atol=1e-10, rtol=0)
        prefix.require(
            result["selected_absolute_endpoint"] == expected["selected_absolute_endpoint"], "Actual absolute endpoint"
        )
    return result


def checker_action(expected):
    prefix.require(
        all(name not in sys.modules for name in FORBIDDEN_CHECKER_MODULES), "Checker must not import candidate policies"
    )

    def action(_q, _limits, _jacobian, _previous, offset):
        return {"command": np.asarray(expected[offset]["command"], dtype=float), "newly_selected": False}

    return action


def reconcile_stop(continuation, expected_stop, *, start_step, current_margin, prefix_inner_loss=False, horizon=8):
    """Verify physical stops and explicitly preserve non-recomputed decisions."""
    reason = None if expected_stop is None else expected_stop.get("reason")
    if reason not in ("quality_no_witness", "quality_witness_rejected", "observed_joint_buffer_loss"):
        prefix.require(continuation["stop"] == expected_stop, "Replayed physical stop differs")
        continuation["decision_stop_recorded_not_recomputed"] = False
        return continuation
    periods = continuation["periods"]
    prefix.require(
        continuation["stop"] is None
        and type(horizon) is int
        and horizon in (8, 16)
        and len(periods) < horizon
        and expected_stop.get("at_step") == start_step + len(periods)
        and expected_stop.get("next_action_dispatched") is False,
        "Recorded decision stop must follow all replayed commands",
    )
    if reason == "observed_joint_buffer_loss":
        prior_loss = bool(periods and periods[-1]["inner_below_buffer_samples"] > 0) or (
            not periods and prefix_inner_loss
        )
        prefix.require(prior_loss or current_margin < 0.05, "Buffer stop must be observable in the replay")
        np.testing.assert_allclose(current_margin, expected_stop["minimum_current_margin_rad"], atol=1e-10, rtol=0)
    continuation["stop"] = expected_stop.copy()
    continuation["decision_stop_recorded_not_recomputed"] = reason != "observed_joint_buffer_loss"
    continuation["buffer_stop_verified_from_replay"] = reason == "observed_joint_buffer_loss"
    return continuation


def check_prefix_inner(actual, expected):
    """Independently replay the 25 samples immediately preceding the branch."""
    for key in ("prefix_step", "below_buffer_samples", "outside_samples"):
        prefix.require(actual[key] == expected[key], f"Replayed final prefix inner field differs: {key}")
    np.testing.assert_allclose(actual["minimum_margin_rad"], expected["minimum_margin_rad"], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(actual["limits"], expected["limits"])
    prefix.require(len(actual["samples"]) == len(expected["samples"]) == 25, "Both complete final prefix periods")
    for key in ("q", "qvel", "time"):
        np.testing.assert_allclose(
            [sample[key] for sample in actual["samples"]],
            [sample[key] for sample in expected["samples"]],
            atol=1e-10,
            rtol=0,
        )


def reconstruct_gripper(case, result):
    """Rebuild only completed history before passing one controller to the branch."""
    from s2a.research.gripper.causal_loaded_history import reconstruct_loaded_history
    from s2a.research.gripper.loaded_gripper_transaction import LoadedGripperTransaction
    from scripts.simulation.runtime.reconstruct_causal_gripper_history import read_chunk_prefix

    item, n = case["gripper_inputs"], case["spec"]["prefix_steps"]
    chunks = read_chunk_prefix(prefix.checked(item["chunks_ref"]), n // 8)
    parameters = prefix.read(item["parameters_ref"])
    controller, data = reconstruct_loaded_history(
        case["rows"],
        chunks,
        case["tool"],
        parameters,
        velocity_envelope=case["controller"]["gripper_velocity_envelope"],
        gripper_config=case["controller"]["gripper_config"],
        progress=lambda progress: result.update(gripper_reconstruction_progress=progress),
    )
    registered = prefix.read(case["gripper_result_ref"])["data"]
    prefix.require(data == registered, "Reconstructed gripper state and full historical decisions")
    result["gripper_history"] = data
    return LoadedGripperTransaction(controller)


def make_feedback(env, case, selected, gripper, receipt, *, max_steps=8):
    """Restore only the actual last completed arm and gripper history."""
    from s2a.research.joint_control.preview.completed_history import CompletedHistory, restore_recomputed_history
    from s2a.research.joint_control.preview.loaded_gripper_feedback import LoadedGripperFeedback
    from s2a.research.joint_control.preview.quality_feedback_refiner import QualityFeedbackRefiner
    from s2a.research.joint_control.preview.refined_feedback import RefinedPreviewFeedback
    from s2a.research.response_models.causal.causal_inner_error import CausalInnerError

    n, row = case["spec"]["prefix_steps"], case["rows"][-1]
    coupled_record = row["coupled_response_history"]
    prefix.require(
        row["step"] == n - 1 and coupled_record["completed_periods_after"] == n, "Completed causal arm history"
    )
    completed = CompletedHistory(
        previous_command=row["command"],
        coupled_innovation=coupled_record["innovation_after"],
        tracking_innovation=row["tracking_history_after"],
        last_plan=row["tracking_refinement"]["planned_commands"],
        completed_periods=n,
    )
    _q, _limits, jacobian = current_joint_site_state(env)
    _mass, velocity, site = current_robot_dynamics(env, jacobian)
    twist = jacobian @ velocity
    inner = CausalInnerError()
    quality = QualityFeedbackRefiner(inner, receipt, start_step=n, start_cursor=selected["cursor"])
    coupled, tracking, difference = restore_recomputed_history(
        completed,
        row["qualified_forward_input"]["x_site_pose"][0],
        row["qualified_forward_input"]["x_site_twist"][0],
        coupled_record["commit"]["nominal_response"],
        row["tracking_refinement"]["nominal_response"],
        site,
        twist,
        refiner=quality,
    )
    site_from_tcp = inverse(np.asarray(case["controller"]["source_from_site"])) @ case["tool"]
    arm = RefinedPreviewFeedback(
        env,
        coupled,
        tracking,
        selected["targets"] @ inverse(site_from_tcp),
        selected["anchor"] @ inverse(site_from_tcp),
        site_from_tcp,
        selected["cursor"],
        completed.previous_command,
        case["controller"]["contract"],
        receipt,
    )
    receipt["initial_innovation_recompute_max_difference"] = difference
    return LoadedGripperFeedback(
        env,
        arm,
        gripper,
        inner,
        selected["targets"],
        selected["openings"],
        selected["cursor"],
        site_from_tcp,
        n,
        receipt,
        initial_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
        max_steps=max_steps,
    )


def worker(phase, index, mode):
    prefix.environment()
    key = branch_key(index, mode)
    directory = ROOT / phase / key
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = binding.load_inputs()
    prefix.require(
        input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen branch inputs and sources",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    selected = binding.selected_case(case, mode)
    expected, expected_stop = None, None
    if phase == "checker":
        original = prefix.read(registration["producer_branch_refs"][key])
        prefix.require(original["complete"] is original["passed"] is True, "Closed producer branch")
        expected = original["runtime"]["continuation"]["periods"]
        expected_stop = original["runtime"]["continuation"]["stop"]
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "case": case["spec"],
        "branch": mode,
        "automatic_retry": False,
    }
    try:
        gripper = None if phase == "checker" else reconstruct_gripper(case, result)
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            policy = None
            if phase == "checker":
                check_prefix_inner(receipt["final_prefix_inner"], original["runtime"]["final_prefix_inner"])
                action = checker_action(expected)
            else:
                policy = make_feedback(env, case, selected, gripper, receipt)
                action = policy.select

            def observed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                prefix.require(
                    period["step"] == case["spec"]["prefix_steps"] + offset, "Just-completed physical endpoint"
                )
                history = None
                if policy is not None:
                    history = policy.complete(
                        actual_env, command, offset, [item["q"] for item in period["actual_inner"]]
                    )
                expected_score = None if expected is None else expected[offset]["completed_feedback"]["scores"]
                scores = score_completed(
                    actual_env, case, selected, mode, offset, period["tcp_error"], expected=expected_score
                )
                return {"scores": scores, "policy_history": history, "policy_history_recomputed": policy is not None}

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=case["spec"]["prefix_steps"],
                previous_command=case["commands"][-1],
                targets=selected["fixed_targets"],
                tool=case["tool"],
                target_object=case["spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=observed,
                progress=progress,
            )
            if phase == "checker":
                q, limits, _jac = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                return reconcile_stop(
                    outcome,
                    expected_stop,
                    start_step=case["spec"]["prefix_steps"],
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                )
            return outcome

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 32 == 0 or completed >= case["spec"]["prefix_steps"]:
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "case_index": index,
                            "branch": mode,
                            "control_completed": completed,
                            "new_commands": receipt.get("newly_selected_commands", 0),
                            "native_steps": receipt["counters"]["mj_step"],
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )

        try:
            runtime = run_prefix_runtime(
                case["config"],
                case["record"],
                case["record_xml"],
                case["commands"],
                case["saved"],
                case["receipts"],
                directory / "runtime",
                inputs["asset_refs"],
                case["controller"],
                continuation=continuation,
                continuation_budget=8,
                registered_seed=42,
                capture_final_prefix_inner=True,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result.update(complete=True, passed=True, runtime=runtime)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute(phase):
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA source identity")
    inputs, sources = binding.load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry preserves registered inputs",
    )
    producer_refs = None
    if phase == "checker":
        producer = prefix.read(prefix.reference(ROOT / "producer/result.json"))
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == 4,
            "Four closed producer branches",
        )
        producer_refs = {
            branch_key(*branch): ref for branch, ref in zip(BRANCHES, producer["branch_refs"], strict=True)
        }
    phase_dir = ROOT / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    registration_ref = publish(
        phase_dir / "registration.json",
        {
            "phase": phase,
            "inputs": input_receipt(inputs),
            "source_refs": sources,
            "producer_branch_refs": producer_refs,
        },
    )
    specs = [
        {
            "key": branch_key(index, mode),
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_causal_event_feedback",
                "--worker",
                phase,
                str(index),
                mode,
            ],
            "log": phase_dir / branch_key(index, mode) / "worker.log",
        }
        for index, mode in BRANCHES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(phase_dir / "workers.json", lifecycle)
    available = [
        prefix.reference(phase_dir / branch_key(*branch) / "result.json")
        for branch in BRANCHES
        if (phase_dir / branch_key(*branch) / "result.json").is_file()
    ]
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "registration_ref": registration_ref,
        "workers_ref": workers_ref,
        "available_branch_refs": available,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 4
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "All four registered workers closed and were reaped",
        )
        branch_refs = [prefix.reference(phase_dir / branch_key(*branch) / "result.json") for branch in BRANCHES]
        branches = [prefix.read(ref) for ref in branch_refs]
        prefix.require(
            all(
                item["complete"] is item["passed"] is True and item["runtime"]["cleanup_complete"] is True
                for item in branches
            ),
            "All bounded branch executions and cleanup closed",
        )
        control = sum(item["runtime"]["control_steps_completed"] for item in branches)
        native = sum(item["runtime"]["counters"]["mj_step"]["completed"] for item in branches)
        models = sum(item["runtime"]["counters"]["model_construct"]["completed"] for item in branches)
        contexts = sum(item["runtime"]["counters"]["context"]["completed"] for item in branches)
        selected = sum(item["runtime"]["newly_selected_commands"] for item in branches)
        prefix.require(
            control <= 1024 and native == 25 * control and native <= 25600 and models == contexts == 12,
            "Registered physical cost limits",
        )
        counts = {
            "control_steps": control,
            "native_steps": native,
            "model_initializations": models,
            "contexts": contexts,
            "newly_selected_commands": selected,
        }
        if phase == "producer":
            histories = [item["gripper_history"] for item in branches]
            proposals = [item["runtime"].get("gripper_proposals", {}) for item in branches]
            calls = sum(item["planner_calls"] for item in histories) + sum(
                item.get("planner_calls", 0) for item in proposals
            )
            sequences = sum(item["candidate_sequences"] for item in histories) + sum(
                item.get("candidate_sequences", 0) for item in proposals
            )
            prefix.require(calls <= 1024 and sequences <= 6718464, "Registered gripper solver budget")
            refiner_calls = sum(item["runtime"].get("refiner_calls", {}).get("attempted", 0) for item in branches)
            refiner_evals = sum(
                item["runtime"].get("quality_refiner", {}).get("baseline_response_evaluations", 0) for item in branches
            )
            quality_calls = sum(
                item["runtime"].get("quality_refiner", {}).get("quality_calls_attempted", 0) for item in branches
            )
            quality_evals = sum(
                item["runtime"].get("quality_refiner", {}).get("quality_response_evaluations", 0) for item in branches
            )
            prefix.require(
                refiner_calls <= 32 and refiner_evals <= 15360 and quality_calls <= 32 and quality_evals <= 5120,
                "Registered arm solver budget",
            )
            counts.update(
                gripper_planner_calls=calls,
                gripper_candidate_sequences=sequences,
                refiner_calls=refiner_calls,
                refiner_response_evaluations=refiner_evals,
                quality_calls=quality_calls,
                quality_response_evaluations=quality_evals,
            )
        else:
            prefix.require(selected == 0, "Checker only replays actual actions")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged sources and inputs")
        result.update(complete=True, passed=True, sources_unchanged=True, branch_refs=branch_refs, counts=counts)
    finally:
        print(publish(phase_dir / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "inputs": input_receipt(binding.load_inputs()),
                    "model_initializations": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.producer or args.checker:
        execute("producer" if args.producer else "checker")
    else:
        phase, index, mode = args.worker
        prefix.require(phase in ("producer", "checker"), "Registered phase")
        worker(phase, int(index), mode)


if __name__ == "__main__":
    main()
