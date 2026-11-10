"""Execute and independently replay one causal old-to-latest reference switch."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_causal_event_feedback as binding
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_event_feedback as base
from scripts.simulation.runtime import run_causal_event_long_feedback as long

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/causal_event_switch_feedback_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/causal_event_switch_feedback_20261110.md"
QA = REPO / "outputs/qa/causal_event_switch_feedback_20261110/result.json"
DRY = REPO / "outputs/causal_event_switch_feedback_dry_20261110/result.json"
ROOT = REPO / "outputs/causal_event_switch_feedback_20261110"
LONG_PRODUCER = {
    "path": str(REPO / "outputs/causal_event_long_feedback_20261110/producer/result.json"),
    "sha256": "269e73cfafb5b030a50c1767344b9369722c03f54bd26052f97f536504742223",
}
LONG_CHECKER = {
    "path": str(REPO / "outputs/causal_event_long_feedback_20261110/checker/result.json"),
    "sha256": "0b691f5a3f625547746d166755bded13f7e06d86f40235b3027e507af627173d",
}


def source_refs():
    return long.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__),
            CONFIG,
            PLAN,
            REPO / "tests/research/native_runtime/test_causal_event_switch_feedback.py",
        )
    ]


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    prefix.require(
        config
        == {
            "schema": "s2a.causal-event-switch-feedback.v1",
            "case_index": 5,
            "case_id": "task1_init0_s42_terminal058",
            "prefix_steps": 360,
            "old_step": 344,
            "task_id": 9,
            "target": "wine_bottle_1",
            "seed": 42,
            "continuation_steps": 16,
            "switch_after_steps": 8,
            "latest_cursor_at_switch": 8,
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "comparison_atol": 1e-10,
            "comparison_rtol": 0,
            "max_control_steps_per_phase": 376,
            "max_native_steps_per_phase": 9400,
            "max_models_per_phase": 3,
            "max_contexts_per_phase": 3,
            "max_refiner_calls_per_phase": 16,
            "max_refiner_response_evaluations_per_phase": 7680,
            "max_quality_calls_per_phase": 16,
            "max_quality_response_evaluations_per_phase": 2560,
            "max_gripper_planner_calls_per_phase": 376,
            "max_gripper_candidate_sequences_per_phase": 2466936,
            "max_workers": 1,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered post-closure reference-switch protocol",
    )
    return ref, config


def load_inputs():
    config_ref, config = read_config()
    producer, checker = prefix.read(LONG_PRODUCER), prefix.read(LONG_CHECKER)
    prefix.require(
        producer["complete"]
        is producer["passed"]
        is producer["sources_unchanged"]
        is checker["complete"]
        is checker["passed"]
        is checker["sources_unchanged"]
        is True
        and len(producer["branch_refs"]) == len(checker["branch_refs"]) == 2,
        "Both prior 16-period branches and independent replay closed",
    )
    inputs = long.load_inputs()
    case = inputs["case"]
    prefix.require(
        all(
            case["spec"][key] == config[key]
            for key in ("case_index", "case_id", "prefix_steps", "old_step", "task_id", "target")
        ),
        "Exact registered task and prefix",
    )
    refs = dict(inputs["source_sha256"])
    refs.update(
        {
            config_ref["path"]: config_ref["sha256"],
            LONG_PRODUCER["path"]: LONG_PRODUCER["sha256"],
            LONG_CHECKER["path"]: LONG_CHECKER["sha256"],
        }
    )
    prior_branches = {}
    for mode, index in (("latest", 0), ("remembered", 1)):
        producer_ref, checker_ref = producer["branch_refs"][index], checker["branch_refs"][index]
        original, replayed = prefix.read(producer_ref, refs), prefix.read(checker_ref, refs)
        prefix.require(
            original["complete"] is original["passed"] is replayed["complete"] is replayed["passed"] is True
            and original["case"]["case_id"] == replayed["case"]["case_id"] == config["case_id"]
            and original["branch"] == replayed["branch"] == mode
            and original["runtime"]["continuation"]["steps_completed"]
            == replayed["runtime"]["continuation"]["steps_completed"]
            == 16,
            "Original paired fixed-reference branches completed and replayed",
        )
        prior_branches[mode] = {"producer_ref": producer_ref, "checker_ref": checker_ref}
    old, latest = (binding.selected_case(case, mode, horizon=16) for mode in ("remembered", "latest"))
    prefix.require(old["cursor"] == 16 and latest["cursor"] == 0, "Original generated reference cursors")
    return {
        "config_ref": config_ref,
        "config": config,
        "case": case,
        "source_sha256": refs,
        "asset_refs": inputs["asset_refs"],
        "cv2_admission": inputs["cv2_admission"],
        "prior_branches": prior_branches,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "case": inputs["case"]["spec"],
        "source_sha256": inputs["source_sha256"],
        "asset_refs": inputs["asset_refs"],
        "prior_branches": inputs["prior_branches"],
    }


def switched_targets(case):
    old = binding.selected_case(case, "remembered", horizon=16)
    latest = binding.selected_case(case, "latest", horizon=16)
    targets = np.concatenate((old["fixed_targets"][:8], latest["fixed_targets"][8:16]))
    openings = np.concatenate((old["fixed_openings"][:8], latest["fixed_openings"][8:16]))
    prefix.require(
        targets.shape == (16, 4, 4) and openings.shape == (16,), "Sixteen same-time paired reference endpoints"
    )
    return old, latest, targets, openings


def control_history(policy):
    arm, past = policy.arm_policy, policy.inner_history.snapshot()
    return {
        "previous_command": arm.previous.copy(),
        "coupled_innovation": arm.coupled.innovation.copy(),
        "tracking_innovation": arm.tracking.history.innovation.copy(),
        "arm_completed_periods": arm.coupled.completed_periods,
        "tracking_completed_periods": arm.tracking.history.completed_periods,
        "inner_completed_periods": past["completed_periods"],
        "inner_position_error_rad": past["inner_position_error_rad"].copy(),
        "gripper": copy.deepcopy(policy.gripper.state()),
    }


def switch_reference(policy, latest, receipt):
    """Rebind only at the completed boundary and prove histories unchanged."""
    before = control_history(policy)
    receipt["reference_switch"] = {"attempted": True, "completed": False}
    detail = policy.rebind_reference(
        latest["targets"], latest["openings"], latest["anchor"], offset=8, next_cursor=8, generated_step=360
    )
    after = control_history(policy)
    for key in ("previous_command", "coupled_innovation", "tracking_innovation", "inner_position_error_rad"):
        np.testing.assert_array_equal(after[key], before[key])
    for key in ("arm_completed_periods", "tracking_completed_periods", "inner_completed_periods", "gripper"):
        prefix.require(after[key] == before[key], f"Reference switch changed completed history: {key}")
    receipt["reference_switch"].update(completed=True, detail=detail, unchanged_history=after)


def worker(phase):
    prefix.environment()
    directory = ROOT / phase / "switch"
    registration = prefix.read(prefix.reference(ROOT / phase / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        input_receipt(inputs) == registration["inputs"] and source_refs() == registration["source_refs"],
        "Frozen switch sources and inputs",
    )
    case = inputs["case"]
    old, latest, targets, _openings = switched_targets(case)
    earlier = prefix.read(inputs["prior_branches"]["remembered"]["producer_ref"])
    earlier_periods = earlier["runtime"]["continuation"]["periods"]
    expected, expected_stop = None, None
    if phase == "checker":
        producer = prefix.read(registration["producer_branch_ref"])
        prefix.require(producer["complete"] is producer["passed"] is True, "Closed switch producer")
        expected, expected_stop = (
            producer["runtime"]["continuation"]["periods"],
            producer["runtime"]["continuation"]["stop"],
        )
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "case": case["spec"],
        "automatic_retry": False,
    }
    try:
        gripper = None if phase == "checker" else base.reconstruct_gripper(case, result)
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            base.check_prefix_inner(receipt["final_prefix_inner"], earlier["runtime"]["final_prefix_inner"])
            start_position = long.target_position(env, case["spec"]["target"])
            receipt["target_start_position"], receipt["previous_verified_steps"] = start_position, 0
            policy = None
            if phase == "checker":
                np.testing.assert_allclose(
                    start_position, producer["runtime"]["target_start_position"], atol=1e-10, rtol=0
                )
                action = base.checker_action(expected)
            else:
                policy = base.make_feedback(env, case, old, gripper, receipt, max_steps=16)

                def action(q, limits, jacobian, previous, offset):
                    if offset == 8 and not policy.last_inner_loss:
                        switch_reference(policy, latest, receipt)
                    return policy.select(q, limits, jacobian, previous, offset)

            def observed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                prefix.require(period["step"] == 360 + offset, "Just-completed switch endpoint")
                history = None
                if policy is not None:
                    history = policy.complete(
                        actual_env, command, offset, [item["q"] for item in period["actual_inner"]]
                    )
                mode = "remembered" if offset < 8 else "latest"
                expected_score = None
                if expected is not None:
                    expected_score = expected[offset]["completed_feedback"]["scores"]
                elif offset < 8:
                    expected_score = earlier_periods[offset]["completed_feedback"]["scores"]
                scores = base.score_completed(
                    actual_env, case, old, mode, offset, period["tcp_error"], expected=expected_score
                )
                if offset < 8:
                    long.check_previous_period(period, earlier_periods[offset])
                    receipt["previous_verified_steps"] += 1
                position = long.target_position(actual_env, case["spec"]["target"])
                height = float(position[2] - start_position[2])
                if expected is not None:
                    recorded = expected[offset]["completed_feedback"]
                    prefix.require(recorded["reference_used"] == mode, "Replayed reference switch time")
                    np.testing.assert_allclose(position, recorded["object_position"], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(height, recorded["object_height_change_m"], atol=1e-10, rtol=0)
                return {
                    "scores": scores,
                    "reference_used": mode,
                    "object_position": position,
                    "object_height_change_m": height,
                    "policy_history": history,
                    "policy_history_recomputed": policy is not None,
                }

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=360,
                previous_command=case["commands"][-1],
                targets=targets,
                tool=case["tool"],
                target_object=case["spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=observed,
                progress=progress,
                continuation_limit=16,
            )
            prefix.require(
                receipt["previous_verified_steps"] == 8, "Earlier completed old-reference prefix must reproduce"
            )
            if phase == "checker":
                q, limits, _jac = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                outcome = base.reconcile_stop(
                    outcome,
                    expected_stop,
                    start_step=360,
                    current_margin=margin,
                    prefix_inner_loss=receipt["final_prefix_inner"]["below_buffer_samples"] > 0,
                    horizon=16,
                )
                outcome["switch_decision_not_recomputed"] = True
            return outcome

        def progress(receipt):
            completed = receipt.get("control_steps_completed", 0)
            if completed % 64 == 0 or completed >= 360:
                print(
                    json.dumps(
                        {
                            "phase": phase,
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
                continuation_budget=16,
                continuation_limit=16,
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
        "Frozen switch CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry preserves switch inputs",
    )
    producer_ref = None
    if phase == "checker":
        producer = prefix.read(prefix.reference(ROOT / "producer/result.json"))
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True,
            "Switch producer closed",
        )
        producer_ref = producer["branch_ref"]
    phase_dir = ROOT / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    registration_ref = publish(
        phase_dir / "registration.json",
        {"phase": phase, "inputs": input_receipt(inputs), "source_refs": sources, "producer_branch_ref": producer_ref},
    )
    specs = [
        {
            "key": "switch",
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_causal_event_switch_feedback",
                "--worker",
                phase,
            ],
            "log": phase_dir / "switch/worker.log",
        }
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(phase_dir / "workers.json", lifecycle)
    branch_path = phase_dir / "switch/result.json"
    result = {
        "complete": False,
        "passed": False,
        "control_qualified": False,
        "phase": phase,
        "registration_ref": registration_ref,
        "workers_ref": workers_ref,
        "available_branch_ref": prefix.reference(branch_path) if branch_path.is_file() else None,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 1
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "Registered switch worker closed and was reaped",
        )
        branch_ref = prefix.reference(branch_path)
        branch = prefix.read(branch_ref)
        runtime = branch["runtime"]
        prefix.require(
            branch["complete"] is branch["passed"] is runtime["cleanup_complete"] is True
            and runtime["previous_verified_steps"] == 8,
            "Earlier eight periods and cleanup verified",
        )
        control, native = runtime["control_steps_completed"], runtime["counters"]["mj_step"]["completed"]
        models, contexts = (
            runtime["counters"]["model_construct"]["completed"],
            runtime["counters"]["context"]["completed"],
        )
        selected = runtime["newly_selected_commands"]
        prefix.require(
            control <= 376 and native == 25 * control and native <= 9400 and models == contexts == 3,
            "Registered switch physical budget",
        )
        counts = {
            "control_steps": control,
            "native_steps": native,
            "model_initializations": models,
            "contexts": contexts,
            "newly_selected_commands": selected,
        }
        if phase == "producer":
            history, proposal = branch["gripper_history"], runtime.get("gripper_proposals", {})
            calls = history["planner_calls"] + proposal.get("planner_calls", 0)
            sequences = history["candidate_sequences"] + proposal.get("candidate_sequences", 0)
            refiner_calls = runtime.get("refiner_calls", {}).get("attempted", 0)
            refiner_evals = runtime.get("quality_refiner", {}).get("baseline_response_evaluations", 0)
            quality_calls = runtime.get("quality_refiner", {}).get("quality_calls_attempted", 0)
            quality_evals = runtime.get("quality_refiner", {}).get("quality_response_evaluations", 0)
            prefix.require(
                calls <= 376
                and sequences <= 2466936
                and refiner_calls <= 16
                and refiner_evals <= 7680
                and quality_calls <= 16
                and quality_evals <= 2560,
                "Registered switch solver budgets",
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
            prefix.require(selected == 0, "Switch checker only replays actual actions")
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(REPO)["sources"] == before, "Unchanged switch sources and inputs"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_ref=branch_ref, counts=counts)
    finally:
        print(publish(phase_dir / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", choices=("producer", "checker"))
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
                    "inputs": input_receipt(load_inputs()),
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
        worker(args.worker)


if __name__ == "__main__":
    main()
