"""Audit eight-point quality at the actual replan input, without dispatch."""

from pathlib import Path

import numpy as np

from s2a.research.joint_control.tracking.velocity_warning_inputs import current_recorded_decision
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_replan_quality_feedback as quality
from scripts.simulation.runtime import run_replan_reference_switch as switch
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as prior
from scripts.tracking.trajectory_search import audit_velocity_warning_full_horizon as warning

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/replan_full_horizon_quality_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/replan_full_horizon_quality_20261110.md"
QA = REPO / "outputs/qa/replan_full_horizon_quality_20261110/result.json"
DRY = REPO / "outputs/replan_full_horizon_quality_dry_20261110/result.json"
OUTPUT = REPO / "outputs/replan_full_horizon_quality_20261110"
PRODUCER = {
    "path": str(quality.ROOT / "producer/result.json"),
    "sha256": "d6541e22dece07ca214b02f4df7b5b81ce613a632802116d933df8032deee63f",
}
CHECKER = {
    "path": str(quality.ROOT / "checker/result.json"),
    "sha256": "0fca5c925a4477efa213f9578d2802eb4fd3d6724b35ea4d6123bd7916eb16f2",
}
LABELS = ("hard_seed", "hard_warm", "restore_seed", "restore_warm")
CASES = (2, 6)


def source_refs():
    paths = (
        Path(__file__),
        CONFIG,
        PLAN,
        REPO / "tests/research/native_runtime/test_replan_full_horizon_quality_inputs.py",
    )
    return quality.source_refs() + warning.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config
        == {
            "schema": "s2a.replan-full-horizon-quality.v1",
            "cases": [2, 6],
            "sampling_seed": 43,
            "start_step": 16,
            "reference_chunk_index": 2,
            "reference_cursor": 0,
            "branches": list(LABELS),
            "horizon": 8,
            "control_dt_s": 0.05,
            "inner_dt_s": 0.002,
            "joint_buffer_rad": 0.05,
            "braking_time_s": 0.1,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "max_models": 2,
            "max_forward_queries": 2,
            "max_optimizer_calls": 8,
            "max_optimizer_responses": 2560,
            "max_independent_endpoint_queries": 1600,
            "max_independent_transition_steps": 20800,
            "worker_timeout_seconds": 1800,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered replan eight-point diagnostic",
    )
    inputs = quality.load_inputs()
    refs = inputs["source_sha256"]
    refs.update(
        {str(CONFIG): config_ref["sha256"], PRODUCER["path"]: PRODUCER["sha256"], CHECKER["path"]: CHECKER["sha256"]}
    )
    prefix.checked(prior.OSMESA)
    refs[prior.OSMESA["path"]] = prior.OSMESA["sha256"]
    branches = {}
    for ref in (quality.BASE_PRODUCER, quality.BASE_CHECKER, PRODUCER, CHECKER):
        owner = prefix.read(ref)
        prefix.require(
            owner["complete"] is owner["passed"] is owner["sources_unchanged"] is True, "Closed producer/checker"
        )
        for branch_ref in owner["branch_refs"]:
            data = prefix.read(branch_ref)
            refs[branch_ref["path"]] = branch_ref["sha256"]
            key = (data["phase"], data["case_index"], data["branch"])
            prefix.require(key not in branches and data["complete"] is data["passed"] is True, "Unique closed branch")
            branches[key] = data
    for case in inputs["cases"]:
        index = case["spec"]["case_index"]
        prefix.require(index in CASES and case["spec"]["prefix_steps"] == 16, "Exact original replan endpoint")
        related = [
            branches[(phase, index, branch)] for branch in ("new", "quality_new") for phase in ("producer", "checker")
        ]
        initial = np.asarray(related[0]["runtime"]["runtime_initial_joint"])
        for item in related:
            np.testing.assert_array_equal(item["runtime"]["runtime_initial_joint"], initial)
            prefix.require(
                item["runtime"]["initial_joint_historical_value_available"] is False,
                "Historical initial joint remains unknown",
            )
        baseline, quality_branch = related[0], related[2]
        baseline_first = baseline["runtime"]["continuation"]["periods"][0]
        quality_first = quality_branch["runtime"]["continuation"]["periods"][0]
        prefix.require(baseline_first["step"] == quality_first["step"] == 16, "Only the current decision is used")
        row = prefix.read_recorded_prefix(case["physical_folder"] / "steps.jsonl", 17, 600)[-1]
        decision = current_recorded_decision(row, step=16, previous_command=case["commands"][-1])
        np.testing.assert_allclose(baseline_first["decision"]["refined_plan"], decision["plan"], atol=1e-8, rtol=0)
        warm = np.asarray(quality_first["decision"]["refined_plan"], dtype=float)
        prefix.require(warm.shape == (8, 6), "One current independently audited warm plan")
        np.testing.assert_array_equal(warm[0], np.asarray(quality_first["command"])[:6])
        selected = switch.selected_case(case, "new")
        case.update(
            targets=selected["targets"],
            anchor=selected["anchor"],
            intervention_spec={"cursor": 0, "chunk_index": 2},
            initial_joint=initial,
            decision=decision,
            first_point_warm=warm,
            feedback_periods=[{"step": 16, "decision": {"refined_plan": decision["plan"]}}],
        )
    return inputs


def candidate_for_audit(label, solved):
    """A relaxed candidate is joint-safe only until all pose caps pass separately."""
    prefix.require(label in LABELS, "Registered solver branch")
    if label.startswith("restore_"):
        return solved.get("best_joint_safe")
    return solved if solved.get("witness_found") is True else None
