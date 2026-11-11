"""Run and independently replay fixed-reference feedback from admitted resets."""

import argparse
import copy
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.development_continuation import current_joint_site_state, run_continuation
from scripts.simulation.runtime import prepare_reset_feedback as preparation
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/joint_control/reset_start_feedback_20261111.json"
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/reset_start_feedback_20261111.md"
OUTPUT = ROOT / "outputs/reset_start_feedback_20261111"
DRY = ROOT / "outputs/reset_start_feedback_dry_20261111/result.json"
QA = ROOT / "outputs/qa/reset_start_feedback_20261111/result.json"
MODULE = "scripts.simulation.runtime.run_reset_start_feedback"
BRANCHES = tuple((i, "feedback") for i in range(12))
CHECKER_FORBIDDEN = (
    "s2a.research.joint_control.preview.reset_feedback_policy",
    "s2a.research.joint_control.preview.refined_feedback",
    "s2a.research.joint_control.preview.quality_feedback_refiner",
    "s2a.research.joint_control.preview.shared_shift_fallback",
    "s2a.research.joint_control.preview.loaded_gripper_feedback",
    "s2a.research.joint_control.tracking.loaded_velocity_control",
    "s2a.research.gripper.loaded_gripper_transaction",
)


def require_no_simulator_import():
    prefix.require(
        not any(name in sys.modules for name in ("mujoco", "OpenGL", "robosuite", "libero")),
        "Simulator must not load before the native OSMesa setup",
    )


def validate_config(config):
    expected = dict(
        schema="s2a.reset-start-feedback.v1",
        case_indices=list(range(12)),
        target_objects=list(preparation.TARGETS),
        continuation_steps=8,
        control_dt_s=0.05,
        inner_dt_s=0.002,
        position_cap_m=0.01,
        rotation_cap_rad=0.05,
        joint_buffer_rad=0.05,
        arm_rate_limit=0.25,
        max_workers=12,
        max_models_per_phase=36,
        max_contexts_per_phase=36,
        max_control_steps_per_phase=96,
        max_native_steps_per_phase=2400,
        max_refiner_calls_per_phase=96,
        max_refiner_responses_per_phase=46080,
        max_quality_calls_per_phase=96,
        max_quality_responses_per_phase=30720,
        max_gripper_planner_calls_per_phase=96,
        max_gripper_candidate_sequences_per_phase=629856,
        max_audit_endpoint_queries_per_phase=19200,
        max_audit_transition_steps_per_phase=249600,
        worker_timeout_seconds=3600,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    if config != expected or config.get("automatic_retry") is not False:
        raise ValueError("The fixed three-task reset feedback roster and budgets are required")
    for key in (
        "continuation_steps",
        "max_workers",
        "max_models_per_phase",
        "max_contexts_per_phase",
        "max_control_steps_per_phase",
        "max_native_steps_per_phase",
        "max_refiner_calls_per_phase",
        "max_refiner_responses_per_phase",
        "max_quality_calls_per_phase",
        "max_quality_responses_per_phase",
        "max_gripper_planner_calls_per_phase",
        "max_gripper_candidate_sequences_per_phase",
        "max_audit_endpoint_queries_per_phase",
        "max_audit_transition_steps_per_phase",
        "worker_timeout_seconds",
        "world_model_calls",
        "fresh_cases",
    ):
        if type(config[key]) is not int:
            raise ValueError("Feedback budgets must be explicit integers")


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__).resolve(),
        Path(preparation.__file__).resolve(),
        ROOT / "scripts/simulation/runtime/prepare_reset_development.py",
        ROOT / "scripts/simulation/runtime/registered_feedback_phase.py",
        ROOT / "src/s2a/research/native_runtime/reset_feedback_runtime.py",
        ROOT / "src/s2a/research/native_runtime/development_reset_runtime.py",
        ROOT / "src/s2a/research/native_runtime/reset_canvas_threads.py",
        ROOT / "src/s2a/research/native_runtime/reset_endpoint.py",
        ROOT / "src/s2a/research/native_runtime/development_continuation.py",
        ROOT / "src/s2a/research/joint_control/preview/reset_feedback_policy.py",
        ROOT / "src/s2a/research/joint_control/preview/reset_history.py",
        ROOT / "src/s2a/research/joint_control/preview/refined_feedback.py",
        ROOT / "src/s2a/research/joint_control/preview/quality_feedback_refiner.py",
        ROOT / "src/s2a/research/joint_control/preview/shared_shift_fallback.py",
        ROOT / "src/s2a/research/joint_control/preview/loaded_gripper_feedback.py",
        ROOT / "src/s2a/research/gripper/loaded_gripper_transaction.py",
        ROOT / "src/s2a/research/joint_control/tracking/loaded_velocity_control.py",
        ROOT / "tests/research/native_runtime/test_reset_start_feedback.py",
        ROOT / "tests/research/joint_control/preview/test_reset_feedback_policy.py",
        ROOT / "tests/research/native_runtime/test_reset_feedback_runtime.py",
        ROOT / "tests/research/native_runtime/test_prepare_reset_feedback.py",
    )
    return [prefix.reference(path) for path in paths]


def load_inputs(index=None, *, verify_assets=True):
    require_no_simulator_import()
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    validate_config(config)
    inputs = preparation.load_inputs(index, verify_assets=verify_assets)
    refs = {config_ref["path"]: config_ref["sha256"]}
    for ref in (
        preparation.ADMISSION,
        preparation.AUDIT,
        inputs["boot_ref"],
        inputs["migrated_ref"],
        inputs["mapped_ref"],
        inputs["source_preflight_ref"],
        inputs["study_inputs"]["config_ref"],
        *inputs["study_inputs"]["refs"].values(),
    ):
        refs[ref["path"]] = ref["sha256"]
    for ref in inputs["study_inputs"]["asset_refs"]:
        refs[ref["path"]] = ref["sha256"]
    refs.update(inputs["asset_refs"])
    for case in inputs["cases"]:
        refs.update(case["original_source_sha256"])
        for ref in (case["original_receipt_ref"], case["gripper_parameters_ref"], case["reset_admission_ref"]):
            refs[ref["path"]] = ref["sha256"]
    inputs.update(feedback_config=config, feedback_config_ref=config_ref, source_sha256=refs)
    require_no_simulator_import()
    return inputs


def input_receipt(inputs):
    return dict(base=preparation.input_receipt(inputs), feedback_config_ref=inputs["feedback_config_ref"])


def receipt_for_case(receipt, index):
    value = copy.deepcopy(receipt)
    base = value["base"]
    selected = [entry for entry in base["cases"] if entry["case_index"] == index]
    raw = [entry for entry in base["base"]["cases"] if entry["case"]["case_index"] == index]
    prefix.require(len(selected) == len(raw) == 1, "One matching registered case required")
    base["cases"], base["base"]["cases"] = selected, raw
    return value


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "One fixed feedback branch required")
    return f"case_{index:02d}_{mode}"


def checker_isolation():
    prefix.require(
        all(name not in sys.modules for name in CHECKER_FORBIDDEN), "Checker must not import candidate policies"
    )


def checker_action(expected):
    checker_isolation()

    def action(_q, _limits, _jacobian, _previous, offset):
        return {"command": np.asarray(expected[offset]["command"], dtype=float), "newly_selected": False}

    return action


def measured_gripper(env, reference):
    """Score actual aperture and motor only after a completed physical period."""
    robot, data = env.robots[0], env.sim.data
    q = np.asarray(data.qpos)[robot._ref_gripper_joint_pos_indexes].copy()
    velocity = np.asarray(data.qvel)[robot._ref_gripper_joint_vel_indexes].copy()
    motor = np.asarray(robot.gripper.current_action, dtype=float).copy()
    prefix.require(
        q.shape == velocity.shape == motor.shape == (2,) and np.isfinite([*q, *velocity, *motor, reference]).all(),
        "Finite actual gripper state and reference required",
    )
    opening = float(np.clip((q[0] - q[1] - 0.001) / 0.079, 0, 1))
    return dict(
        opening=opening,
        opening_reference=float(reference),
        opening_error=opening - float(reference),
        qpos=q,
        qvel=velocity,
        motor_target=motor,
    )


def compare_score(actual, expected):
    for key in ("opening", "opening_reference", "opening_error", "qpos", "qvel", "motor_target"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=0)


def reconcile_stop(outcome, expected_stop, *, current_margin):
    """Preserve solver stops as reported, and independently verify physical stops."""
    if expected_stop is None or expected_stop.get("reason") == "actual_inner_joint_limit_violation":
        prefix.require(outcome["stop"] == expected_stop, "The actual physical stop must match")
        outcome["decision_stop_recorded_not_recomputed"] = False
        return outcome
    reason, periods = expected_stop.get("reason"), outcome["periods"]
    prefix.require(
        reason in ("quality_no_witness", "quality_witness_rejected", "observed_joint_buffer_loss")
        and outcome["stop"] is None
        and len(periods) < 8
        and expected_stop.get("at_step") == len(periods)
        and expected_stop.get("next_action_dispatched") is False,
        "A reported decision stop must follow the actually replayed commands",
    )
    if reason == "observed_joint_buffer_loss":
        prefix.require(
            current_margin < 0.05 or bool(periods and periods[-1]["inner_below_buffer_samples"]),
            "The buffer stop must be observable in replay",
        )
        np.testing.assert_allclose(current_margin, expected_stop["minimum_current_margin_rad"], atol=1e-10, rtol=0)
    outcome["stop"] = expected_stop.copy()
    outcome["decision_stop_recorded_not_recomputed"] = reason != "observed_joint_buffer_loss"
    return outcome


def summarize_phase(branches, phase, config):
    prefix.require(phase in ("producer", "checker") and len(branches) == 12, "All twelve branches required")
    totals = dict(
        models=0,
        contexts=0,
        control_steps=0,
        native_steps=0,
        newly_selected_commands=0,
        refiner_calls=0,
        refiner_responses=0,
        quality_calls=0,
        quality_responses=0,
        gripper_planner_calls=0,
        gripper_candidate_sequences=0,
        audit_endpoint_queries=0,
        audit_transition_steps=0,
        short_window_passes=0,
    )
    per_case = []
    for index, branch in enumerate(branches):
        runtime = branch["runtime"]
        prefix.require(
            branch["complete"] is branch["passed"] is True
            and branch["phase"] == phase
            and branch["case"]["case_index"] == index
            and runtime["reset_observation_verified"] is True
            and runtime["reset_image_match"] is runtime["cleanup_complete"] is True
            and runtime["previous_saved_command"] is None
            and runtime["canvas_thread_scope"] == {"before": 1, "requested": 2, "inside": 2, "after": 1},
            "Every branch must begin from its exact admitted reset",
        )
        c, periods = runtime["counters"], runtime["continuation"]["periods"]
        n = len(periods)
        prefix.require(
            n <= 8
            and runtime["control_steps_completed"] == n
            and c["mj_step"]["completed"] == c["mj_step"]["attempted"] == 25 * n
            and c["mj_step2"]["attempted"] == 0
            and c["model_construct"]["completed"] == c["context"]["completed"] == 3,
            "Actual model and physical budgets must match",
        )
        selected = runtime["newly_selected_commands"]
        prefix.require(selected == (n if phase == "producer" else 0), "Candidate selections must be phase-specific")
        prefix.require(
            phase != "checker" or branch["candidate_modules_unloaded"] is True,
            "Checker must not import candidate policies",
        )
        successful = n == 8 and all(p["pose_caps_satisfied"] and p["inner_below_buffer_samples"] == 0 for p in periods)
        counts = runtime.get("quality_refiner", {})
        gripper = runtime.get("gripper_proposals", {})
        values = dict(
            models=c["model_construct"]["completed"],
            contexts=c["context"]["completed"],
            control_steps=n,
            native_steps=c["mj_step"]["completed"],
            newly_selected_commands=selected,
            refiner_calls=counts.get("baseline_calls_attempted", 0),
            refiner_responses=counts.get("baseline_response_evaluations", 0),
            quality_calls=counts.get("quality_calls_attempted", 0),
            quality_responses=counts.get("quality_response_evaluations", 0),
            gripper_planner_calls=gripper.get("planner_calls", 0),
            gripper_candidate_sequences=gripper.get("candidate_sequences", 0),
            audit_endpoint_queries=counts.get("endpoint_queries_charged", 0),
            audit_transition_steps=counts.get("transition_steps_charged", 0),
            short_window_passes=int(successful),
        )
        for key, value in values.items():
            totals[key] += value
        per_case.append(
            dict(
                case=branch["case"],
                completed=n,
                short_window_passed=bool(successful),
                stop=runtime["continuation"]["stop"],
                counts=values,
            )
        )
    for key, limit in (
        ("models", "max_models_per_phase"),
        ("contexts", "max_contexts_per_phase"),
        ("control_steps", "max_control_steps_per_phase"),
        ("native_steps", "max_native_steps_per_phase"),
        ("refiner_calls", "max_refiner_calls_per_phase"),
        ("refiner_responses", "max_refiner_responses_per_phase"),
        ("quality_calls", "max_quality_calls_per_phase"),
        ("quality_responses", "max_quality_responses_per_phase"),
        ("gripper_planner_calls", "max_gripper_planner_calls_per_phase"),
        ("gripper_candidate_sequences", "max_gripper_candidate_sequences_per_phase"),
        ("audit_endpoint_queries", "max_audit_endpoint_queries_per_phase"),
        ("audit_transition_steps", "max_audit_transition_steps_per_phase"),
    ):
        prefix.require(totals[key] <= config[limit], f"The {key} budget was exceeded")
    return dict(totals=totals, cases=per_case)


def dry_run():
    prefix.environment()
    inputs = load_inputs(verify_assets=True)
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=input_receipt(inputs),
                source_refs=source_refs(),
                native_models=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                automatic_retry=False,
            ),
        ),
        flush=True,
    )


def worker(phase, index, mode):
    directory = OUTPUT / phase / branch_key(index, mode)
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        case={"case_index": index},
        automatic_retry=False,
        world_model_calls=0,
        fresh_cases=0,
    )
    try:
        temp, cache = directory / "tmp", directory / "cache"
        temp.mkdir()
        cache.mkdir()
        os.environ.update(TMPDIR=str(temp), XDG_CACHE_HOME=str(cache), NUMBA_CACHE_DIR=str(cache / "numba"))
        tempfile.tempdir = None
        prefix.environment()
        registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))
        inputs = load_inputs(index, verify_assets=False)
        prefix.require(
            registration["phase"] == phase
            and receipt_for_case(registration["inputs"], index) == input_receipt(inputs)
            and registration["source_refs"] == source_refs(),
            "Frozen branch input and source identity required",
        )
        if phase == "checker":
            checker_isolation()
            producer = prefix.read(registration["producer_branch_refs"][branch_key(index, mode)])
            prefix.require(producer["complete"] is producer["passed"] is True, "A closed producer branch is required")
            expected = producer["runtime"]["continuation"]["periods"]
            expected_stop = producer["runtime"]["continuation"]["stop"]
        else:
            expected, expected_stop = None, None
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        require_no_simulator_import()
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure
        from s2a.research.native_runtime.reset_feedback_runtime import ResetFeedbackRuntime

        case = inputs["cases"][0]
        adapter = ResetFeedbackRuntime(
            expected=case["expected"],
            expected_receipt=case["expected_receipt"],
            chunk=case["chunk"],
            controller=case["controller"],
            tool=case["tool"],
        )

        def continuation(env, terminal, observers, receipt, *, progress=None):
            if phase == "producer":
                from s2a.research.joint_control.preview.reset_feedback_policy import build_reset_policy, policy_inputs

                policy, previous = build_reset_policy(env, policy_inputs(case, terminal), receipt)

                def action(q, limits, jacobian, prior, offset):
                    start = time.monotonic()
                    decision = policy.select(q, limits, jacobian, prior, offset)
                    decision["total_decision_seconds"] = time.monotonic() - start
                    return decision
            else:
                checker_isolation()
                policy, previous, action = None, np.zeros(7), checker_action(expected)

            def completed(actual_env, command, offset):
                period = receipt["continuation_partial"]["periods"][-1]
                history = (
                    None
                    if policy is None
                    else policy.complete(
                        actual_env, command, offset, [sample["q"] for sample in period["actual_inner"]]
                    )
                )
                score = measured_gripper(actual_env, case["chunk"]["openings"][offset])
                if expected is not None:
                    compare_score(score, expected[offset]["completed_feedback"]["score"])
                return dict(score=score, policy_history=history, policy_history_recomputed=policy is not None)

            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=0,
                previous_command=previous,
                targets=np.asarray(case["chunk"]["effective_targets_tcp"][:8]),
                tool=case["tool"],
                target_object=case["target_object"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
            )
            if phase == "checker":
                q, limits, _ = current_joint_site_state(env)
                margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                return reconcile_stop(outcome, expected_stop, current_margin=margin)
            return outcome

        try:
            runtime = adapter.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                directory / "runtime",
                inputs["asset_refs"],
                registered_seed=case["case"]["sampling_seed"],
                continuation=continuation,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        result.update(
            complete=True,
            passed=True,
            case=case["case"],
            runtime=runtime,
            candidate_modules_unloaded=all(name not in sys.modules for name in CHECKER_FORBIDDEN),
        )
        prefix.require(phase != "checker" or result["candidate_modules_unloaded"] is True, "Checker isolation required")
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def execute(phase):
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["all_tests_passed"] is True
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"],
        "Complete source-matched CPU QA required",
    )
    registered_feedback_phase.execute_phase(
        phase,
        root=OUTPUT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module=MODULE,
        branch_key=branch_key,
        load_inputs=load_inputs,
        input_receipt=input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="feedback_config",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "INDEX", "MODE"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.phase:
        execute(args.phase)
    else:
        worker(args.worker[0], int(args.worker[1]), args.worker[2])


if __name__ == "__main__":
    main()
