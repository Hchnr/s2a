"""Bind one cross-task fallback to prior strict feedback and physical replays."""

from pathlib import Path

from scripts.simulation.runtime import prepare_replan_shared_shift as replan
from scripts.simulation.runtime import prepare_task2_earlier_preview as task2_inputs
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_replan_shared_shift_feedback as replan_runner
from scripts.simulation.runtime import run_task2_shared_shift_feedback as task2_runner

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/model_safe_tracking_fallback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/model_safe_tracking_fallback_20261111.md"
PRIOR = {
    "replan": {
        "producer": {
            "path": str(REPO / "outputs/replan_shared_shift_feedback_20261111/producer/result.json"),
            "sha256": "9899bb96b0e5a48d6fa2b15957df650539861c3f056cf422254f7ba336f550c8",
        },
        "checker": {
            "path": str(REPO / "outputs/replan_shared_shift_feedback_20261111/checker/result.json"),
            "sha256": "11096d434279fdda7a08be59bcf4a7d16e733ee6c93da57e34a2a12a9822aa8f",
        },
    },
    "task2": replan.TASK2,
}
GROUPS = {"replan": (2, 6), "task2": (10, 11)}


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    expected = {
        "schema": "s2a.model-safe-tracking-fallback.v1",
        "replan_case_indices": [2, 6],
        "task2_case_indices": [10, 11],
        "seed": 43,
        "continuation_steps": 8,
        "control_dt_s": 0.05,
        "inner_dt_s": 0.002,
        "joint_buffer_rad": 0.05,
        "position_cap_m": 0.01,
        "rotation_cap_rad": 0.05,
        "fallback_position_limit_m": 0.02,
        "fallback_rotation_limit_rad": 0.1,
        "arm_rate_limit": 0.25,
        "braking_time_s": 0.1,
        "quality_response_budget": 320,
        "max_optimizer_iterations": 120,
        "max_feasibility_iterations": 60,
        "max_start_lp_per_call": 2,
        "max_start_lp_per_phase": 64,
        "max_control_steps_per_phase": 176,
        "max_native_steps_per_phase": 4400,
        "max_models_per_phase": 12,
        "max_contexts_per_phase": 12,
        "max_refiner_calls_per_phase": 32,
        "max_refiner_response_evaluations_per_phase": 15360,
        "max_quality_calls_per_phase": 32,
        "max_quality_response_evaluations_per_phase": 10240,
        "max_shifted_rollouts_per_phase": 64,
        "max_audit_endpoint_queries_per_phase": 6400,
        "max_audit_transition_steps_per_phase": 83200,
        "max_workers": 4,
        "worker_timeout_seconds": 1800,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    prefix.require(config == expected, "Exact cross-task fallback protocol and budgets")
    return ref, config


def source_refs():
    return replan_runner.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            REPO / "tests/research/native_runtime/test_model_safe_tracking_fallback_inputs.py",
        )
    ]


def policy_case(group, case):
    if group == "replan":
        return replan.policy_case(case)
    if group == "task2":
        return task2_inputs.policy_case(case)
    raise ValueError("One registered development case group required")


def load_inputs():
    config_ref, config = read_config()
    datasets = {"replan": replan.load_inputs(), "task2": task2_runner.load_inputs()}
    prefix.require(
        datasets["replan"]["asset_refs"] == datasets["task2"]["asset_refs"]
        and datasets["replan"]["cv2_admission"] == datasets["task2"]["cv2_admission"],
        "Both groups use the same pinned native assets and runtime admission",
    )
    refs = {}
    for dataset in datasets.values():
        for path, digest in dataset["source_sha256"].items():
            prefix.require(path not in refs or refs[path] == digest, "Shared input identities cannot conflict")
            refs[path] = digest
    prefix.checked(config_ref, refs)
    cases, prior_branches = {}, {}
    for group, indices in GROUPS.items():
        group_cases = datasets[group]["cases"]
        prefix.require(tuple(case["spec"]["case_index"] for case in group_cases) == indices, "Fixed case order")
        for case in group_cases:
            index = case["spec"]["case_index"]
            policy_case(group, case)
            cases[index] = {"group": group, "case": case}
        previous_runner = replan_runner if group == "replan" else task2_runner
        for phase in ("producer", "checker"):
            outcome = prefix.read(PRIOR[group][phase], refs)
            prefix.require(
                outcome["phase"] == phase
                and outcome["complete"] is outcome["passed"] is outcome["sources_unchanged"] is True
                and outcome["control_qualified"] is False
                and len(outcome["branch_refs"]) == len(indices),
                "The prior strict feedback and independent replay must be closed",
            )
            registration = prefix.read(outcome["registration_ref"], refs)
            prefix.require(registration["source_refs"] == previous_runner.source_refs(), "Prior strict source identity")
            for index, ref in zip(indices, outcome["branch_refs"], strict=True):
                item = prefix.read(ref, refs)
                prefix.require(
                    item["case"]["case_index"] == index
                    and item["complete"] is item["passed"] is True
                    and (phase != "checker" or item["candidate_modules_unloaded"] is True),
                    "Each strict development branch and its replay must match",
                )
                prior_branches.setdefault(index, {})[phase] = ref
    prefix.require(tuple(sorted(cases)) == (2, 6, 10, 11), "All three tasks and four registered cases")
    return {
        "config_ref": config_ref,
        "config": config,
        "cases": cases,
        "datasets": datasets,
        "asset_refs": datasets["replan"]["asset_refs"],
        "cv2_admission": datasets["replan"]["cv2_admission"],
        "source_sha256": refs,
        "prior_branch_refs": prior_branches,
    }


def input_receipt(inputs):
    return {
        "config_ref": inputs["config_ref"],
        "cases": {str(index): entry["case"]["spec"] for index, entry in inputs["cases"].items()},
        "asset_refs": inputs["asset_refs"],
        "source_sha256": inputs["source_sha256"],
        "prior_refs": PRIOR,
        "prior_branch_refs": {str(index): refs for index, refs in inputs["prior_branch_refs"].items()},
    }
