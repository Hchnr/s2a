"""Run two real-feedback branches from the first combined warning."""

import json
from pathlib import Path

import numpy as np

from s2a.geometry import pose_error
from s2a.research.joint_control.tracking.combined_warning_inputs import first_combined_warning
from scripts.simulation.runtime import compare_multicase_emergency_retreat as previous
from scripts.simulation.runtime import reconstruct_early_refiner as reconstruction
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_early_quality_feedback as quality

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/combined_warning_feedback_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/combined_warning_feedback_20261110.md"
QA = REPO / "outputs/qa/combined_warning_feedback_20261110/result.json"
DRY = REPO / "outputs/combined_warning_feedback_dry_20261110/result.json"
ROOT = REPO / "outputs/combined_warning_feedback_20261110"
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


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "src/s2a/research/joint_control/tracking/combined_warning_inputs.py",
        REPO / "src/s2a/research/joint_control/limits/quality_warning.py",
        REPO / "tests/research/joint_control/tracking/test_combined_warning_inputs.py",
        REPO / "tests/research/native_runtime/test_combined_warning_feedback_inputs.py",
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
        history = reconstruction.completed_from_prefix(rows, case["commands"])
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
        prefix.require(q.shape == (7,) and limits.shape == (7, 2) and np.isfinite(q).all(), "Finite current joints")
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


def reconcile_checker_stop(continuation, expected_stop, *, mode, start_step):
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
        and len(periods) < 8
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
