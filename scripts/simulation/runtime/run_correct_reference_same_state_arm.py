"""Replay two fixed arm sequences from one verified correct-reference state."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.correct_reference_continuation import run_correct_reference_continuation
from s2a.research.native_runtime.correct_reference_reset import CorrectReferenceResetRuntime
from s2a.research.native_runtime.development_continuation import run_continuation
from s2a.research.native_runtime.development_prefix_replay import SAMPLE_FIELDS
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from scripts.simulation.runtime import audit_reset_multitask_live_recovery as comparison
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import prepare_correct_reference_same_state_arm as preparation
from scripts.simulation.runtime import registered_feedback_phase as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_correct_reference as previous
from scripts.simulation.runtime import run_reset_start_feedback as original

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/correct_reference_same_state_arm_20261116"
DRY = ROOT / "outputs/correct_reference_same_state_arm_dry_20261116.json"
QA = ROOT / "outputs/qa/correct_reference_same_state_arm_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_correct_reference_same_state_arm"
BRANCHES = ((3, "candidate_actions"), (3, "original_arm"))


def source_refs():
    return preparation.cohort.source_refs() + [
        prefix.reference(path)
        for path in (
            preparation.CONFIG,
            preparation.PLAN,
            Path(preparation.__file__).resolve(),
            Path(__file__).resolve(),
            ROOT / "scripts/simulation/runtime/registered_feedback_phase.py",
            ROOT / "tests/research/native_runtime/test_correct_reference_same_state_arm_inputs.py",
            ROOT / "tests/research/native_runtime/test_correct_reference_same_state_arm.py",
        )
    ]


def branch_key(index, branch):
    prefix.require((index, branch) in BRANCHES, "One fixed diagnostic branch is required")
    return f"case_{index:02d}_{branch}"


def compare_branchpoints(left, right):
    """Compare physical state, all recorded controller fields, and motor cache."""
    if left["endpoint_receipt"] != right["endpoint_receipt"]:
        raise ValueError("The two branchpoints must have the same endpoint receipt")
    maximum = 0.0
    for name in (*SAMPLE_FIELDS, "gripper_current_action"):
        a, b = np.asarray(left["sample"][name], dtype=float), np.asarray(right["sample"][name], dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Finite matching branchpoint fields required: {name}")
        difference = float(np.max(np.abs(a - b)))
        if difference > 1e-10:
            raise ValueError(f"The branchpoint field differs: {name}")
        maximum = max(maximum, difference)
    return maximum


def dry_run():
    prefix.environment()
    original.require_no_simulator_import()
    inputs = preparation.load_inputs()
    original.checker_isolation()
    print(
        publish_result(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=preparation.input_receipt(inputs),
                source_refs=source_refs(),
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


def worker(phase, index, branch):
    prefix.require(phase in ("producer", "checker"), "One registered phase is required")
    key = branch_key(index, branch)
    directory = OUTPUT / phase / key
    registration = prefix.read(prefix.reference(OUTPUT / phase / "registration.json"))

    def execute():
        inputs = preparation.load_inputs()
        prefix.require(
            registration["inputs"] == preparation.input_receipt(inputs)
            and registration["source_refs"] == source_refs(),
            "The fixed diagnostic inputs and sources must match registration",
        )
        original.checker_isolation()
        prior = None
        if phase == "checker":
            prior = prefix.read(registration["producer_branch_refs"][key])
            prefix.require(prior["complete"] is prior["passed"] is True, "A closed diagnostic producer is required")
        cv2 = inputs["shared"]["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        original.require_no_simulator_import()
        case, reset, config = inputs["case"], inputs["case"]["reset"], inputs["config"]
        adapter = CorrectReferenceResetRuntime(
            expected=reset["expected"],
            expected_receipt=reset["expected_receipt"],
            controller=case["case"]["controller"],
            tool=case["case"]["tool"],
            anchor=reset["anchor"],
            opening=reset["opening"],
        )

        def continuation(env, terminal, observers, receipt, *, progress=None):
            cache = SameStateEndpointCache(observers.endpoint)
            wrapped = SimpleNamespace(
                endpoint=cache, current=observers.current, render_cameras=observers.render_cameras
            )
            replay = run_correct_reference_continuation(
                env,
                terminal,
                wrapped,
                receipt,
                timeline=case["timeline"],
                tool=case["case"]["tool"],
                target_object=config["target_object"],
                phase="checker",
                score_gripper=original.measured_gripper,
                compare_score=original.compare_score,
                recorded=inputs["recorded_prefix"],
                progress=progress,
            )
            prefix.require(
                replay["summary"]["completed_periods"] == config["prefix_periods"]
                and replay["stop"] is None
                and replay["summary"]["inner_below_buffer_samples"] == 0,
                "The exact candidate prefix must reach the safe branchpoint",
            )
            sample, endpoint = cache(env, config["prefix_periods"])
            expected_endpoint = inputs["candidate_periods"][config["prefix_periods"] - 1]["endpoint_receipt"]
            prefix.require(endpoint == expected_endpoint, "The branchpoint must match the original candidate endpoint")
            branchpoint = dict(
                sample={name: np.asarray(sample[name]).copy() for name in (*SAMPLE_FIELDS, "gripper_current_action")},
                endpoint_receipt=endpoint,
            )
            new_terminal = dict(sample=sample, endpoint_receipt=endpoint)
            targets, openings, _ = case["timeline"].window(config["prefix_periods"])
            expected = None if prior is None else prior["runtime"]["continuation"]["intervention"]["periods"]
            sequence = inputs["actions"][branch]

            def choose(q, limits, _jacobian, _previous, offset):
                if phase == "producer":
                    completed = receipt["continuation_partial"]["periods"]
                    margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
                    if margin < config["joint_buffer_rad"] or (
                        completed and completed[-1]["inner_below_buffer_samples"]
                    ):
                        return dict(stop=dict(reason="observed_joint_buffer_loss", minimum_current_margin_rad=margin))
                return dict(command=sequence[offset].copy(), newly_selected=False, fixed_diagnostic_branch=branch)

            def completed(actual_env, _command, offset):
                score = original.measured_gripper(actual_env, openings[offset])
                if expected is not None:
                    original.compare_score(score, expected[offset]["completed_feedback"]["score"])
                return dict(score=score, policy_history=None, policy_history_recomputed=False)

            outcome = run_continuation(
                env,
                new_terminal,
                wrapped,
                receipt,
                start_step=config["prefix_periods"],
                previous_command=inputs["previous_command"],
                targets=targets[: config["intervention_periods"]],
                tool=case["case"]["tool"],
                target_object=config["target_object"],
                choose_action=choose,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=completed,
                progress=progress,
                continuation_limit=config["intervention_periods"],
            )
            if prior is not None:
                reported = prior["runtime"]["continuation"]["intervention"]["stop"]
                if reported is not None and reported["reason"] == "actual_inner_joint_limit_violation":
                    prefix.require(outcome["stop"] == reported, "A physical limit stop must be reproduced")
                else:
                    prefix.require(outcome["stop"] is None, "The checker observed an unexpected physical stop")
                    outcome["stop"] = reported
                compare_branchpoints(branchpoint, prior["runtime"]["continuation"]["branchpoint"])
                replay_max = comparison.compare_periods(outcome["periods"], expected)
            else:
                replay_max = None
            periods = outcome["periods"]
            np.testing.assert_array_equal(
                np.asarray([row["command"] for row in periods]).reshape(-1, 7), sequence[: len(periods)]
            )
            historical_max = None
            if branch == "candidate_actions":
                start = config["prefix_periods"]
                historical_max = comparison.compare_periods(
                    periods, inputs["candidate_periods"][start : start + len(periods)]
                )
            return dict(
                prefix=replay,
                branchpoint=branchpoint,
                intervention=outcome,
                checker_replay_max_difference=replay_max,
                historical_candidate_max_difference=historical_max,
                stop_recomputed=False,
            )

        def progress(receipt):
            print(json.dumps(dict(branch=branch, **previous.progress_event(phase, receipt))), flush=True)

        runtime = adapter.run_feedback(
            case["runtime_config"],
            case["record"],
            case["record_xml"],
            directory / "runtime",
            inputs["shared"]["asset_refs"],
            registered_seed=config["seed"],
            continuation=continuation,
            max_control_steps=config["prefix_periods"] + config["intervention_periods"],
            progress=progress,
        )
        original.checker_isolation()
        return dict(
            complete=True,
            passed=True,
            case_index=index,
            branch=branch,
            runtime=runtime,
            candidate_modules_unloaded=True,
            cv2_admission=cv2_receipt,
            fresh_cases=0,
        )

    managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)


def summarize_phase(branches, phase, config):
    prefix.require(
        len(branches) == 2 and all(row["complete"] is row["passed"] is True for row in branches),
        "Both diagnostic workers must close",
    )
    prefix.require([row["branch"] for row in branches] == [branch for _, branch in BRANCHES], "Branch order must match")
    runtimes = [row["runtime"] for row in branches]
    for item, runtime in zip(branches, runtimes, strict=True):
        intervention = runtime["continuation"]["intervention"]["periods"]
        prefix.require(
            runtime["complete"] is runtime["passed"] is runtime["reset_observation_verified"] is True
            and item["candidate_modules_unloaded"] is True
            and runtime["newly_selected_commands"] == 0
            and runtime["control_steps_completed"]
            == runtime["control_steps_attempted"]
            == config["prefix_periods"] + len(intervention)
            and runtime["counters"]["mj_step"]["completed"]
            == runtime["counters"]["mj_step"]["attempted"]
            == 25 * runtime["control_steps_completed"]
            and runtime["counters"]["model_construct"]["completed"] == runtime["counters"]["context"]["completed"] == 3
            and runtime["cleanup_complete"] is True
            and len(intervention) <= config["intervention_periods"],
            "Every physical branch must close with its exact replay and resource counts",
        )
    branchpoint_max = compare_branchpoints(*(runtime["continuation"]["branchpoint"] for runtime in runtimes))
    counts = dict(
        control_steps=sum(runtime["control_steps_completed"] for runtime in runtimes),
        native_steps=sum(runtime["counters"]["mj_step"]["completed"] for runtime in runtimes),
        model_initializations=sum(runtime["counters"]["model_construct"]["completed"] for runtime in runtimes),
        contexts=sum(runtime["counters"]["context"]["completed"] for runtime in runtimes),
        intervention_steps=[len(runtime["continuation"]["intervention"]["periods"]) for runtime in runtimes],
        branchpoint_max_difference=branchpoint_max,
    )
    prefix.require(
        counts["control_steps"] <= config["max_control_periods_per_phase"]
        and counts["native_steps"] <= config["max_inner_steps_per_phase"]
        and counts["model_initializations"] == counts["contexts"] == config["max_models_per_phase"],
        "The actual diagnostic work must stay inside its registered budget",
    )
    return counts


def phase_run(phase):
    registered.execute_phase(
        phase,
        root=OUTPUT,
        qa_path=QA,
        dry_path=DRY,
        branches=BRANCHES,
        worker_module=MODULE,
        branch_key=branch_key,
        load_inputs=preparation.load_inputs,
        input_receipt=preparation.input_receipt,
        source_refs=source_refs,
        summarize_phase=summarize_phase,
        config_key="config",
    )


def audit():
    """Compare physical outcomes from the same fully observed branchpoint."""
    prefix.environment()
    inputs = preparation.load_inputs()
    producer_ref, checker_ref = (
        prefix.reference(OUTPUT / "producer/result.json"),
        prefix.reference(OUTPUT / "checker/result.json"),
    )
    producer, checker = prefix.read(producer_ref), prefix.read(checker_ref)
    prefix.require(
        producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
        and checker["complete"] is checker["passed"] is checker["sources_unchanged"] is True,
        "Both registered phases must close with unchanged sources",
    )
    produced = [prefix.read(ref) for ref in producer["branch_refs"]]
    checked = [prefix.read(ref) for ref in checker["branch_refs"]]
    branchpoint_max = compare_branchpoints(*(item["runtime"]["continuation"]["branchpoint"] for item in produced))
    replay_max = []
    for left, right in zip(produced, checked, strict=True):
        compare_branchpoints(
            left["runtime"]["continuation"]["branchpoint"], right["runtime"]["continuation"]["branchpoint"]
        )
        replay_max.append(
            comparison.compare_periods(
                left["runtime"]["continuation"]["intervention"]["periods"],
                right["runtime"]["continuation"]["intervention"]["periods"],
            )
        )
    candidate, alternate = [item["runtime"]["continuation"]["intervention"]["periods"] for item in produced]
    common = min(len(candidate), len(alternate))
    prefix.require(common > 0, "At least one actual paired intervention period is required")
    cols = inputs["object_state_columns"]
    a = np.asarray([row["actual_state"] for row in candidate[:common]])[:, cols]
    b = np.asarray([row["actual_state"] for row in alternate[:common]])[:, cols]
    object_diff = np.linalg.norm(a - b, axis=1)
    tcp_diff = np.linalg.norm(
        np.asarray([row["tcp_error"][:3] for row in candidate[:common]])
        - np.asarray([row["tcp_error"][:3] for row in alternate[:common]]),
        axis=1,
    )
    baseline_rows = prefix.read(inputs["identity"]["refs"]["rows"])
    start, target = inputs["config"]["prefix_periods"], inputs["config"]["target_object"]
    historical = np.asarray([row["objects"][target]["position"] for row in baseline_rows[start : start + common]])
    distances_a, distances_b = np.linalg.norm(a - historical, axis=1), np.linalg.norm(b - historical, axis=1)
    results = dict(
        common_periods=common,
        branchpoint_max_difference=branchpoint_max,
        object_position_difference_max_m=float(np.max(object_diff)),
        object_position_difference_final_m=float(object_diff[-1]),
        tcp_position_difference_max_m=float(np.max(tcp_diff)),
        original_object_distance_final_m=dict(
            candidate_actions=float(distances_a[-1]), original_arm=float(distances_b[-1])
        ),
        task_success_periods=dict(
            candidate_actions=sum(row["task_success"] for row in candidate),
            original_arm=sum(row["task_success"] for row in alternate),
        ),
        grasp_periods=dict(
            candidate_actions=sum(row["target_grasp"] for row in candidate),
            original_arm=sum(row["target_grasp"] for row in alternate),
        ),
        replay_max_difference=replay_max,
    )
    print(
        publish_result(
            OUTPUT / "audit.json",
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                producer_ref=producer_ref,
                checker_ref=checker_ref,
                metrics=results,
                world_model_calls=0,
                fresh_cases=0,
                physics_steps=0,
            ),
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.dry:
        dry_run()
    elif args.phase:
        phase_run(args.phase)
    elif args.worker:
        worker(args.worker[0], int(args.worker[1]), args.worker[2])
    else:
        audit()


if __name__ == "__main__":
    main()
