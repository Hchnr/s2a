"""Audit eight-point quality at the actual replan input, without dispatch."""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from s2a.geometry import finite
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.joint_control.tracking.velocity_warning_inputs import current_recorded_decision
from s2a.research.trajectory_search.full_horizon_quality import solve_full_horizon
from scripts.maintenance.run_cpu_tests import snapshot
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


def worst_quality_slack(errors):
    """Worst relative cap violation, independently recomputed from TCP errors."""
    errors = finite(errors, (8, 6))
    return max(
        0.0,
        float(
            np.max(
                np.r_[
                    np.linalg.norm(errors[:, :3], axis=1) / 0.01 - 1,
                    np.linalg.norm(errors[:, 3:], axis=1) / 0.05 - 1,
                ]
            )
        ),
    )


def analyze(case, stats):
    current, site_from_tcp, targets, args, differences = warning.restore_warning_context(case, stats)
    profile = np.zeros((25, 7))
    output, normals = {}, None
    for label in LABELS:
        extra = {"feasibility_restoration": label.startswith("restore_")}
        if label.endswith("warm"):
            extra["initial_guess"] = case["first_point_warm"]
        stats["optimizer_calls_attempted"] += 1
        started = time.monotonic()
        solved = solve_full_horizon(*args, past_inner_error_rad=profile, **extra)
        stats["optimizer_calls_completed"] += 1
        stats["optimizer_responses"] += int(solved["response_evaluations"])
        prefix.require(solved["response_evaluations"] <= 320, "Per-search response budget")
        if normals is None:
            normals = solved["terminal_normals"]
        else:
            np.testing.assert_array_equal(normals, solved["terminal_normals"])
        candidate = candidate_for_audit(label, solved)
        verification = None
        if candidate is not None:
            stats["independent_endpoint_queries_charged"] += 200
            stats["independent_transition_steps_charged"] += 2600
            verification = audit_full_horizon_witness(
                candidate,
                current["mass"],
                current["jacobian"],
                current["contract"]["kp"],
                current["contract"]["kd"],
                current["q"],
                current["velocity"],
                case["initial_joint"],
                current["interface"],
                current["previous"][:6],
                current["site"],
                site_from_tcp,
                targets,
                current["limits"][:, 0],
                current["limits"][:, 1],
                past_inner_error_rad=profile,
                enforce_pose_caps=not label.startswith("restore_"),
            )
            prefix.require(
                verification["passed"] is True
                and verification["independent_endpoint_queries"] == 200
                and verification["independent_transition_steps"] == 2600,
                "Independent joint and TCP model audit",
            )
            slack = worst_quality_slack(verification["tcp_errors"])
            if "quality_slack" in candidate:
                np.testing.assert_allclose(slack, candidate["quality_slack"], atol=1e-8, rtol=0)
            verification["actual_quality_slack"] = slack
        verified_quality = bool(verification is not None and verification["all_pose_caps_satisfied"] is True)
        if solved["witness_found"]:
            prefix.require(verified_quality, "Every model quality witness must independently satisfy all eight caps")
        summary = {key: value for key, value in solved.items() if key not in ("rollout", "best_joint_safe")}
        if candidate is not None:
            summary["audited_candidate"] = {key: value for key, value in candidate.items() if key != "rollout"}
        summary.update(
            independent_verification=verification,
            quality_witness_verified=verified_quality,
            elapsed_seconds=time.monotonic() - started,
        )
        output[label] = summary
    return {
        "case_index": case["spec"]["case_index"],
        "start_step": 16,
        "reference_chunk": case["intervention_spec"],
        "reconstruction_max_difference": differences,
        "historical_initial_joint_available": False,
        "initial_past_error_available": False,
        "candidates": output,
        "scope": "static frozen model; no action dispatch or physical qualification",
    }


def worker(index):
    def interrupted(signum, _frame):
        raise InterruptedError(f"Worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    inputs = load_inputs()
    prefix.require(
        inputs["source_sha256"] == registration["source_sha256"]
        and inputs["asset_refs"] == registration["asset_refs"]
        and source_refs() == registration["source_refs"],
        "Frozen input and source identities",
    )
    case = next(item for item in inputs["cases"] if item["spec"]["case_index"] == index)
    stats = {
        "model_initializations_attempted": 0,
        "model_initializations_completed": 0,
        "forward_queries_attempted": 0,
        "forward_queries_completed": 0,
        "optimizer_calls_attempted": 0,
        "optimizer_calls_completed": 0,
        "optimizer_responses": 0,
        "baseline_rollouts": 0,
        "independent_endpoint_queries_charged": 0,
        "independent_transition_steps_charged": 0,
        "max_qpos_restore_difference": 0.0,
        "physics_steps": 0,
    }
    result = {
        "complete": False,
        "passed": False,
        "case_index": index,
        "stats": stats,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        result["osmesa_ref"] = prior.preload_software_gl(case, inputs["source_sha256"])
        result["analysis"] = analyze(case, stats)
        prefix.require(
            stats["model_initializations_completed"] == 1
            and stats["forward_queries_completed"] == 1
            and stats["baseline_rollouts"] == 1
            and stats["optimizer_calls_completed"] == 4
            and stats["optimizer_responses"] <= 1280
            and stats["independent_endpoint_queries_charged"] <= 800
            and stats["independent_transition_steps_charged"] <= 10400,
            "Per-case budget",
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / f"case_{index:02d}" / "result.json", result), flush=True)


def execute():
    prefix.require(not OUTPUT.exists(), "New output only; no automatic retry")
    dry_ref, qa_ref = prefix.reference(DRY), prefix.reference(QA)
    dry, qa = prefix.read(dry_ref), prefix.read(qa_ref)
    prefix.require(
        dry["complete"] is dry["passed"] is True and dry["models"] == dry["physics_steps"] == 0,
        "Zero-model dry closed",
    )
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "CPU QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Full QA source identity")
    sources, inputs = source_refs(), load_inputs()
    prefix.require(
        dry["source_refs"] == sources
        and dry["source_sha256"] == inputs["source_sha256"]
        and dry["asset_refs"] == inputs["asset_refs"],
        "Dry source and input identity",
    )
    registration = publish(
        OUTPUT / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "producer_ref": PRODUCER,
            "checker_ref": CHECKER,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "cases": list(CASES),
            "branches": list(LABELS),
            "max_models": 2,
            "max_forward_queries": 2,
            "max_optimizer_calls": 8,
            "max_optimizer_responses": 2560,
            "max_independent_endpoint_queries": 1600,
            "max_independent_transition_steps": 20800,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )
    specs = [
        {
            "key": f"case_{index:02d}",
            "argv": [
                sys.executable,
                "-m",
                "scripts.tracking.trajectory_search.audit_replan_full_horizon_quality",
                "--worker",
                str(index),
            ],
            "log": OUTPUT / f"case_{index:02d}" / "worker.log",
        }
        for index in CASES
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(OUTPUT / "workers.json", lifecycle)
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": registration,
        "workers_ref": workers_ref,
        "physics_steps": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 2
            and all(
                worker["returncode"] == 0 and worker["actual_wait"] and worker["reaped"] and not worker["signal_sent"]
                for worker in lifecycle["workers"]
            ),
            "Both workers closed and reaped",
        )
        refs = [prefix.reference(OUTPUT / f"case_{index:02d}" / "result.json") for index in CASES]
        rows = [prefix.read(ref) for ref in refs]
        prefix.require(all(row["complete"] is row["passed"] is True for row in rows), "Complete replan audits")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs() == sources
            and snapshot(REPO)["sources"] == before,
            "Sources and inputs unchanged",
        )
        totals = {
            key: sum(row["stats"][key] for row in rows)
            for key in rows[0]["stats"]
            if key != "max_qpos_restore_difference"
        }
        prefix.require(
            totals["model_initializations_completed"] == 2
            and totals["forward_queries_completed"] == 2
            and totals["baseline_rollouts"] == 2
            and totals["optimizer_calls_completed"] == 8
            and totals["optimizer_responses"] <= 2560
            and totals["independent_endpoint_queries_charged"] <= 1600
            and totals["independent_transition_steps_charged"] <= 20800
            and totals["physics_steps"] == 0,
            "Registered aggregate budgets",
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=refs, totals=totals)
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int)
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        prefix.require(not DRY.exists(), "New dry output only")
        inputs = load_inputs()
        print(
            publish(
                DRY,
                {
                    "complete": True,
                    "passed": True,
                    "source_refs": source_refs(),
                    "source_sha256": inputs["source_sha256"],
                    "asset_refs": inputs["asset_refs"],
                    "cases": [case["spec"]["case_index"] for case in inputs["cases"]],
                    "models": 0,
                    "physics_steps": 0,
                    "world_model_calls": 0,
                    "fresh_cases": 0,
                },
            ),
            flush=True,
        )
    elif args.execute:
        execute()
    else:
        prefix.require(args.worker in CASES, "Registered case")
        worker(args.worker)


if __name__ == "__main__":
    main()
