"""Run two real-feedback branches from the first combined warning."""

import argparse
import json
import signal
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import pose_error
from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.preview.completed_history import CompletedHistory
from s2a.research.joint_control.tracking.combined_warning_inputs import first_combined_warning
from s2a.research.native_runtime.development_continuation import run_continuation
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import compare_multicase_emergency_retreat as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_early_quality_feedback as quality

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/combined_warning_feedback_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/combined_warning_feedback_20261110.md"
QA = REPO / "outputs/qa/combined_warning_feedback_20261110/result.json"
DRY = REPO / "outputs/combined_warning_feedback_dry_20261110/result.json"
ROOT = REPO / "outputs/combined_warning_feedback_20261110"
LONG_CONFIG = REPO / "configs/tracking/joint_control/combined_warning_long_feedback_20261110.json"
LONG_PLAN = REPO / "plans/v0.0.1/tracking/joint_control/combined_warning_long_feedback_20261110.md"
LONG_QA = REPO / "outputs/qa/combined_warning_long_feedback_20261110/result.json"
LONG_DRY = REPO / "outputs/combined_warning_long_feedback_dry_20261110/result.json"
LONG_ROOT = REPO / "outputs/combined_warning_long_feedback_20261110"
PRIOR_PRODUCER = {
    "path": str(ROOT / "producer/result.json"),
    "sha256": "71aeaa44bdb9a2182d87fa808edfda1bc32e2c67162fd0695bbcfd41f64e64a4",
}
PRIOR_CHECKER = {
    "path": str(ROOT / "checker/result.json"),
    "sha256": "95a9d802307d16ee0729a5c1c4634adf1c75ce26e5a768a3b94301a90d6c7441",
}
AUDIT = {
    "path": str(REPO / "outputs/combined_quality_warning_20261110/result.json"),
    "sha256": "e0b92c73c38aa5e5ec2eefe42c31c9efcc586127e91390d6afd5a04577ce9be9",
}
CASES = (
    {
        "case_index": 2,
        "case_id": "task0_init0_s43_raw",
        "prefix_steps": 9,
        "chunk_index": 1,
        "cursor": 1,
        "target": "akita_black_bowl_1",
    },
    {
        "case_index": 6,
        "case_id": "task1_init0_s43_raw",
        "prefix_steps": 9,
        "chunk_index": 1,
        "cursor": 1,
        "target": "wine_bottle_1",
    },
)
BRANCHES = ((2, "original"), (2, "quality"), (6, "original"), (6, "quality"))


def completed_from_prefix(rows, commands):
    """Extract completed history without importing either candidate policy."""
    n = len(commands)
    prefix.require(n > 0 and len(rows) == n and rows[-1]["step"] == n - 1, "Exactly one completed prefix")
    row = rows[-1]
    prefix.require(np.array_equal(np.asarray(row["command"]), commands[-1]), "Recorded previous command mismatch")
    coupled = row["coupled_response_history"]
    prefix.require(coupled["completed_periods_after"] == n, "Coupled completed periods mismatch")
    return CompletedHistory(
        previous_command=row["command"],
        coupled_innovation=coupled["innovation_after"],
        tracking_innovation=row["tracking_history_after"],
        last_plan=row["tracking_refinement"]["planned_commands"],
        completed_periods=n,
    )


def source_refs(*, longer=False):
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/joint_control/tracking/combined_warning_inputs.py",
        REPO / "src/s2a/research/joint_control/limits/quality_warning.py",
        REPO / "tests/research/joint_control/tracking/test_combined_warning_inputs.py",
        REPO / "tests/research/native_runtime/test_combined_warning_feedback_inputs.py",
    )
    if longer:
        paths += (
            LONG_CONFIG,
            LONG_PLAN,
            REPO / "tests/research/native_runtime/test_long_development_continuation.py",
            REPO / "tests/research/native_runtime/test_combined_warning_long_feedback_inputs.py",
        )
    return quality.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    """Bind only a completed prefix and the reference generated at step eight."""
    config_ref = prefix.reference(CONFIG)
    c = prefix.read(config_ref)
    prefix.require(
        c
        == {
            "schema": "s2a.combined-warning-feedback.v1",
            "cases": list(CASES),
            "branches": ["original", "quality"],
            "continuation_steps": 8,
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "braking_time_s": 0.1,
            "absolute_comparison_tolerance": 1e-10,
            "first_decision_tolerance": 1e-8,
            "max_control_steps_per_phase": 68,
            "max_native_steps_per_phase": 1700,
            "max_models_per_phase": 12,
            "max_contexts_per_phase": 12,
            "max_refiner_calls_per_phase": 32,
            "max_refiner_response_evaluations_per_phase": 15360,
            "max_quality_calls_per_phase": 16,
            "max_quality_response_evaluations_per_phase": 2560,
            "max_audit_endpoint_queries_per_phase": 3200,
            "max_audit_transition_steps_per_phase": 41600,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered scope and budgets",
    )
    audit = prefix.read(AUDIT)
    prefix.require(
        audit["complete"] is audit["passed"] is audit["source_and_input_unchanged"] is audit["reaped"] is True,
        "Closed previous combined-warning audit",
    )
    audit_worker = prefix.read(audit["worker_ref"])
    prefix.require(
        audit_worker["complete"] is audit_worker["passed"] is True and audit_worker["endpoints"] == 7200,
        "Completed twelve-case audit worker",
    )
    inputs = prefix.load_inputs({2: 9, 6: 9})
    refs = inputs["source_sha256"]
    refs.update({str(CONFIG): config_ref["sha256"], AUDIT["path"]: AUDIT["sha256"]})
    refs[audit["worker_ref"]["path"]] = audit["worker_ref"]["sha256"]
    review = prefix.read(prefix.REVIEW)
    source_map = review["source_and_artifact_sha256"]
    for spec in CASES:
        index, n = spec["case_index"], spec["prefix_steps"]
        case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
        prefix.require(case["spec"]["case_id"] == spec["case_id"], "Original case identity")
        audit_case_ref = audit_worker["case_refs"][index]
        prefix.checked(audit_case_ref)
        refs[audit_case_ref["path"]] = audit_case_ref["sha256"]
        folder = Path(review["cases"][index]["mechanical"]["case"])
        prefix.require(folder.is_relative_to(REPO / "outputs"), "Reviewed local development directory")
        chunk_path, tool_path, rows_path = (
            folder / "chunks.json",
            Path(case["config"]["tool_contract"]),
            folder / "steps.jsonl",
        )
        for path in (chunk_path, tool_path, rows_path):
            ref = prefix.reference(path)
            prefix.require(source_map.get(ref["path"]) == ref["sha256"], "Reviewed original input identity")
            refs[ref["path"]] = ref["sha256"]
        rows = prefix.read_recorded_prefix(rows_path, n, 600)
        tool = np.asarray(json.loads(tool_path.read_text())["source_from_tcp"], dtype=float)
        targets_recorded = np.asarray([row["desired_tcp"] for row in rows], dtype=float)
        measured_tcp = np.asarray(case["saved"]["source_pose"])[1:] @ tool
        errors = pose_error(targets_recorded, measured_tcp)
        np.testing.assert_allclose(errors, [row["tcp_error"] for row in rows], atol=1e-10, rtol=0)
        first = first_combined_warning(case["saved"]["sim_states"], errors, case["record_xml"])
        prefix.require(first["endpoint"] == n, "Exactly the first combined warning")
        chunk = previous.read_array_item(chunk_path, spec["chunk_index"])
        targets, anchor = np.asarray(chunk["effective_targets_tcp"]), np.asarray(chunk["anchor_tcp"])
        prefix.require(
            chunk["step"] == 8
            and chunk["step"] + spec["cursor"] == n
            and targets.shape == (40, 4, 4)
            and anchor.shape == tool.shape == (4, 4)
            and np.isfinite(targets).all()
            and np.isfinite(anchor).all()
            and np.isfinite(tool).all(),
            "The one finite reference already generated at step eight",
        )
        prefix.require(
            rows[-1]["chunk_index"] == spec["chunk_index"] and rows[-1]["chunk_physical_step"] == 0,
            "The previously completed decision used this chunk's first target",
        )
        np.testing.assert_allclose(rows[-1]["desired_tcp"], targets[0], atol=1e-10, rtol=0)
        history = completed_from_prefix(rows, case["commands"])
        old = rows[-1]
        case.update(
            intervention_spec=spec,
            physical_folder=folder,
            targets=targets,
            anchor=anchor,
            fixed_targets=targets[1:9].copy(),
            tool=tool,
            first_warning=first,
            completed_history=history,
            previous_evidence={
                "site": old["qualified_forward_input"]["x_site_pose"][0],
                "twist": old["qualified_forward_input"]["x_site_twist"][0],
                "coupled_prediction": old["coupled_response_history"]["commit"]["nominal_response"],
                "tracking_prediction": old["tracking_refinement"]["nominal_response"],
            },
        )
    return inputs


def load_long_inputs():
    """Extend only the current generated reference, preserving every original gate."""
    config_ref = prefix.reference(LONG_CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config
        == {
            "schema": "s2a.combined-warning-long-feedback.v1",
            "cases": [2, 6],
            "sampling_seed": 43,
            "prefix_steps": 9,
            "chunk_index": 1,
            "cursor": 1,
            "branches": ["original", "quality"],
            "continuation_steps": 16,
            "control_dt_s": 0.05,
            "joint_buffer_rad": 0.05,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "braking_time_s": 0.1,
            "max_control_steps_per_phase": 100,
            "max_native_steps_per_phase": 2500,
            "max_models_per_phase": 12,
            "max_contexts_per_phase": 12,
            "max_refiner_calls_per_phase": 64,
            "max_refiner_response_evaluations_per_phase": 30720,
            "max_quality_calls_per_phase": 32,
            "max_quality_response_evaluations_per_phase": 5120,
            "max_audit_endpoint_queries_per_phase": 6400,
            "max_audit_transition_steps_per_phase": 83200,
            "worker_timeout_seconds": 1800,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered longer feedback scope",
    )
    for ref in (PRIOR_PRODUCER, PRIOR_CHECKER):
        result = prefix.read(ref)
        prefix.require(
            result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and len(result["branch_refs"]) == 4,
            "The earlier eight-step producer and checker are closed",
        )
    inputs = load_inputs()
    inputs["source_sha256"].update(
        {
            str(LONG_CONFIG): config_ref["sha256"],
            PRIOR_PRODUCER["path"]: PRIOR_PRODUCER["sha256"],
            PRIOR_CHECKER["path"]: PRIOR_CHECKER["sha256"],
        }
    )
    for case in inputs["cases"]:
        prefix.require(case["targets"].shape == (40, 4, 4), "The current generated reference contains sixteen targets")
        case["fixed_targets"] = case["targets"][1:17].copy()
    return inputs


def first_decision_comparison(selected, row):
    """Compare a completed calculation with the original current decision."""
    command = np.asarray(selected["command"], dtype=float)
    old_command = np.asarray(row["command"], dtype=float)
    plan = np.asarray(selected["refined_plan"], dtype=float)
    old_plan = np.asarray(row["tracking_refinement"]["planned_commands"], dtype=float)
    prefix.require(
        command.shape == old_command.shape == (7,) and plan.shape == old_plan.shape == (8, 6), "Decision shapes"
    )
    differences = {
        "arm_command": float(np.max(np.abs(command[:6] - old_command[:6]))),
        "plan": float(np.max(np.abs(plan - old_plan))),
    }
    same_route = selected["route"] == row["tracking_refinement"]["route"]
    return {
        "max_absolute_difference": differences,
        "same_route": same_route,
        "passed": bool(same_route and all(np.isfinite(x) and x <= 1e-8 for x in differences.values())),
        "original_gripper_command": float(old_command[6]),
        "comparison_gripper_command": float(command[6]),
        "original_decision_read_after_recomputation": True,
    }


def guarded_action(choose_action, receipt, *, first_decision_check=None):
    """Stop before a later action once observed inner samples lose the buffer."""

    def select(q, limits, jacobian, previous, offset):
        q, limits = np.asarray(q, dtype=float), np.asarray(limits, dtype=float)
        prefix.require(
            q.shape == (7,)
            and limits.shape == (7, 2)
            and np.isfinite(q).all()
            and np.isfinite(limits).all()
            and np.all(limits[:, 0] < limits[:, 1]),
            "Finite current joints and increasing limits",
        )
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        periods = receipt["continuation_partial"]["periods"]
        prior_loss = bool(periods and periods[-1]["inner_below_buffer_samples"] > 0)
        if margin < 0.05 or prior_loss:
            return {"stop": {"reason": "observed_joint_buffer_loss", "minimum_current_margin_rad": margin}}
        selected = choose_action(q, limits, jacobian, previous, offset)
        if offset == 0 and first_decision_check is not None and "stop" not in selected:
            comparison = first_decision_check(selected)
            receipt["first_decision_reconstruction"] = comparison
            prefix.require(comparison["passed"] is True, "Original first decision reconstruction")
        return selected

    return select


def reconcile_checker_stop(continuation, expected_stop, *, mode, start_step, horizon=8):
    """Verify physical stops and label optimization stops as not re-solved."""
    reason = None if expected_stop is None else expected_stop.get("reason")
    decision_stop = reason in ("quality_no_witness", "quality_witness_rejected", "observed_joint_buffer_loss")
    if not decision_stop:
        prefix.require(continuation["stop"] == expected_stop, "Replayed physical stop differs from producer")
        continuation["decision_stop_recorded_not_recomputed"] = False
        return continuation
    periods = continuation["periods"]
    prefix.require(
        continuation["stop"] is None
        and len(periods) < horizon
        and expected_stop.get("at_step") == start_step + len(periods)
        and expected_stop.get("next_action_dispatched") is False,
        "Decision stop must follow exactly the replayed actions",
    )
    if reason == "observed_joint_buffer_loss":
        prefix.require(periods and periods[-1]["inner_below_buffer_samples"] > 0, "Buffer stop requires observed loss")
    else:
        prefix.require(mode == "quality", "Only quality feedback can stop without a model witness")
    continuation["stop"] = expected_stop.copy()
    continuation["decision_stop_recorded_not_recomputed"] = reason != "observed_joint_buffer_loss"
    continuation["buffer_stop_verified_from_replayed_inner_samples"] = reason == "observed_joint_buffer_loss"
    return continuation


def branch_key(index, mode):
    prefix.require((index, mode) in BRANCHES, "Fixed registered feedback branch")
    return f"case_{index:02d}_{mode}"


def worker(phase, index, mode, *, longer=False):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    root, horizon = (LONG_ROOT, 16) if longer else (ROOT, 8)
    directory = root / phase / branch_key(index, mode)
    registration = prefix.read(prefix.reference(root / phase / "registration.json"))
    inputs = load_long_inputs() if longer else load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs(longer=longer) == registration["source_refs"],
        "Frozen branch inputs and sources",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    expected, expected_stop = None, None
    if phase == "checker":
        original = prefix.read(prefix.reference(root / "producer" / branch_key(index, mode) / "result.json"))
        prefix.require(original["complete"] is original["passed"] is True, "Closed producer branch required")
        expected = original["runtime"]["continuation"]["periods"]
        expected_stop = original["runtime"]["continuation"]["stop"]
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "case_index": index,
        "branch": mode,
        "automatic_retry": False,
    }
    try:
        cv2 = inputs["cv2_admission"]
        result["cv2_admission"] = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
        from s2a.research.native_runtime.development_prefix_runtime import PrefixReplayFailure, run_prefix_runtime

        def continuation(env, terminal, observers, receipt, *, progress=None):
            receipt["preview_calls"] = {"attempted": 0, "completed": 0}
            receipt["refiner_calls"] = {"attempted": 0, "completed": 0}
            receipt["refiner_response_evaluations"] = 0
            on_completed = None
            if phase == "checker":
                action = quality.checker_action(expected)
            elif mode == "original":
                policy = quality.baseline.make_refiner(env, case, receipt)

                def compare_original(selected):
                    row = prefix.read_recorded_prefix(case["physical_folder"] / "steps.jsonl", 10, 600)[-1]
                    prefix.require(row["step"] == 9, "Original current decision identity")
                    return first_decision_comparison(selected, row)

                action = guarded_action(policy.select, receipt, first_decision_check=compare_original)
                on_completed = policy.observe
            else:
                policy, history = quality.make_quality_refiner(env, case, receipt)
                select, on_completed = quality.quality_callbacks(policy, history, receipt, 9)
                action = guarded_action(select, receipt)
            outcome = run_continuation(
                env,
                terminal,
                observers,
                receipt,
                start_step=9,
                previous_command=case["commands"][-1],
                targets=case["fixed_targets"],
                tool=case["tool"],
                target_object=case["intervention_spec"]["target"],
                choose_action=action,
                stop_on_outside=True,
                expected_periods=expected,
                on_completed=on_completed,
                progress=progress,
                continuation_limit=horizon,
            )
            if phase == "checker":
                return reconcile_checker_stop(outcome, expected_stop, mode=mode, start_step=9, horizon=horizon)
            return outcome

        def progress(receipt):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "case": index,
                        "branch": mode,
                        "checked_prefix_endpoints": receipt.get("checked_endpoints", 0),
                        "control_completed": receipt.get("control_steps_completed", 0),
                        "new_commands": receipt.get("newly_selected_commands", 0),
                        "refiner": receipt.get("refiner_calls"),
                        "quality": receipt.get("quality_refiner"),
                        "native_steps": receipt["counters"]["mj_step"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        try:
            receipt = run_prefix_runtime(
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
                continuation_budget=horizon,
                continuation_limit=horizon,
                progress=progress,
            )
        except PrefixReplayFailure as error:
            result["runtime"] = error.receipt
            raise
        if phase == "checker":
            prefix.require(
                quality.QUALITY_MODULE not in sys.modules
                and quality.baseline.POLICY_MODULE not in sys.modules
                and quality.baseline.PREVIEW_MODULE not in sys.modules
                and receipt["newly_selected_commands"] == 0
                and receipt["preview_calls"]["attempted"] == receipt["refiner_calls"]["attempted"] == 0,
                "Checker must replay saved actions without candidate decisions",
            )
        result.update(
            complete=True,
            passed=True,
            runtime=receipt,
            candidate_policy_imported=quality.baseline.POLICY_MODULE in sys.modules,
            preview_module_imported=quality.baseline.PREVIEW_MODULE in sys.modules,
            quality_module_imported=quality.QUALITY_MODULE in sys.modules,
        )
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(directory / "result.json", result), flush=True)


def run_phase(phase, *, longer=False):
    root, dry_path, qa_path, horizon = (LONG_ROOT, LONG_DRY, LONG_QA, 16) if longer else (ROOT, DRY, QA, 8)
    max_controls, max_native = 4 * (9 + horizon), 25 * 4 * (9 + horizon)
    max_refiners, max_quality = 4 * horizon, 2 * horizon
    directory = root / phase
    prefix.require(not directory.exists(), "New phase output only; no automatic retry")
    dry_ref, qa_ref = prefix.reference(dry_path), prefix.reference(qa_path)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    prefix.require(dry["complete"] is dry["passed"] is True and dry["physics_steps"] == 0, "Zero-model dry closed")
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Full QA source identity")
    sources = source_refs(longer=longer)
    inputs = load_long_inputs() if longer else load_inputs()
    prefix.require(
        dry["source_refs"] == sources
        and dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"],
        "Dry source and input identity",
    )
    producer_ref = None
    if phase == "checker":
        producer_ref = prefix.reference(root / "producer/result.json")
        producer = prefix.read(producer_ref)
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == 4,
            "Producer must be complete before replay",
        )
        for ref in producer["branch_refs"]:
            prefix.checked(ref)
    registration = publish(
        directory / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "audit_ref": AUDIT,
            "producer_ref": producer_ref,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "branches": [list(branch) for branch in BRANCHES],
            "max_control_steps": max_controls,
            "max_native_steps": max_native,
            "max_refiner_calls": max_refiners,
            "max_refiner_response_evaluations": 480 * max_refiners,
            "max_quality_calls": max_quality,
            "max_quality_response_evaluations": 160 * max_quality,
            "max_audit_endpoint_queries": 200 * max_quality,
            "max_audit_transition_steps": 2600 * max_quality,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )
    specs = [
        {
            "key": branch_key(index, mode),
            "argv": [
                sys.executable,
                "-m",
                "scripts.simulation.runtime.run_combined_warning_feedback",
                *(["--long"] if longer else []),
                "--worker",
                phase,
                str(index),
                mode,
            ],
            "log": directory / branch_key(index, mode) / "worker.log",
        }
        for index, mode in BRANCHES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(directory / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
        "phase": phase,
        "registration_ref": registration,
        "workers_ref": workers_ref,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 4
            and all(
                worker["returncode"] == 0 and worker["actual_wait"] and worker["reaped"] and not worker["signal_sent"]
                for worker in lifecycle["workers"]
            ),
            "All four workers closed and reaped",
        )
        refs = [prefix.reference(directory / branch_key(index, mode) / "result.json") for index, mode in BRANCHES]
        branches = [prefix.read(ref) for ref in refs]
        prefix.require(all(branch["complete"] is branch["passed"] is True for branch in branches), "Complete branches")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs(longer=longer) == sources
            and snapshot(REPO)["sources"] == before,
            "Sources and inputs unchanged",
        )
        controls = sum(branch["runtime"]["control_steps_completed"] for branch in branches)
        native = sum(branch["runtime"]["counters"]["mj_step"]["completed"] for branch in branches)
        models = sum(branch["runtime"]["counters"]["model_construct"]["completed"] for branch in branches)
        contexts = sum(branch["runtime"]["counters"]["context"]["completed"] for branch in branches)
        refiners = sum(branch["runtime"]["refiner_calls"]["attempted"] for branch in branches)
        evaluations = sum(
            branch["runtime"].get("quality_refiner", {}).get("baseline_response_evaluations", 0)
            if branch["branch"] == "quality"
            else branch["runtime"]["refiner_response_evaluations"]
            for branch in branches
        )
        quality_calls = sum(
            branch["runtime"].get("quality_refiner", {}).get("quality_calls_attempted", 0) for branch in branches
        )
        quality_responses = sum(
            branch["runtime"].get("quality_refiner", {}).get("quality_response_evaluations", 0) for branch in branches
        )
        endpoint_queries = sum(
            branch["runtime"].get("quality_refiner", {}).get("endpoint_queries_charged", 0) for branch in branches
        )
        transition_steps = sum(
            branch["runtime"].get("quality_refiner", {}).get("transition_steps_charged", 0) for branch in branches
        )
        prefix.require(
            controls <= max_controls
            and native == controls * 25
            and native <= max_native
            and models <= 12
            and contexts <= 12
            and refiners <= max_refiners
            and evaluations <= 480 * max_refiners
            and quality_calls <= max_quality
            and quality_responses <= 160 * max_quality
            and endpoint_queries <= 200 * max_quality
            and transition_steps <= 2600 * max_quality,
            "Registered phase budgets",
        )
        if phase == "checker":
            prefix.require(
                refiners == quality_calls == quality_responses == endpoint_queries == transition_steps == 0
                and all(
                    not branch["candidate_policy_imported"]
                    and not branch["preview_module_imported"]
                    and not branch["quality_module_imported"]
                    and branch["runtime"]["newly_selected_commands"] == 0
                    for branch in branches
                ),
                "No independent checker decisions",
            )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            branch_refs=refs,
            control_steps=controls,
            native_steps=native,
            model_initializations=models,
            software_contexts=contexts,
            refiner_calls=refiners,
            refiner_response_evaluations=evaluations,
            quality_calls=quality_calls,
            quality_response_evaluations=quality_responses,
            audit_endpoint_queries_charged=endpoint_queries,
            audit_transition_steps_charged=transition_steps,
            newly_selected_commands=sum(branch["runtime"]["newly_selected_commands"] for branch in branches),
        )
    finally:
        print(publish(directory / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", action="store_true", help="Use the separately registered sixteen-period comparison")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--produce", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--worker", nargs=3, metavar=("PHASE", "CASE", "BRANCH"))
    args = parser.parse_args()
    prefix.environment()
    dry_path = LONG_DRY if args.long else DRY
    if args.dry_run:
        prefix.require(not dry_path.exists(), "New dry output only")
        inputs = load_long_inputs() if args.long else load_inputs()
        print(
            publish(
                dry_path,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(longer=args.long),
                    "source_sha256": inputs["source_sha256"],
                    "asset_refs": inputs["asset_refs"],
                    "cases": [case["intervention_spec"] for case in inputs["cases"]],
                    "first_warnings": [case["first_warning"] for case in inputs["cases"]],
                    "completed_history_periods": [
                        case["completed_history"].completed_periods for case in inputs["cases"]
                    ],
                    "models": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.produce:
        run_phase("producer", longer=args.long)
    elif args.check:
        run_phase("checker", longer=args.long)
    else:
        phase, index, mode = args.worker
        prefix.require(phase in ("producer", "checker"), "Registered phase")
        worker(phase, int(index), mode, longer=args.long)


if __name__ == "__main__":
    main()
