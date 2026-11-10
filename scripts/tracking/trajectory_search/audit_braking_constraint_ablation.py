"""Diagnose an extra model-braking constraint without dispatching actions."""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.joint_control.tracking.full_horizon_quality_audit import audit_full_horizon_witness
from s2a.research.trajectory_search.full_horizon_quality import solve_full_horizon
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as prior
from scripts.tracking.trajectory_search import audit_full_horizon_restoration as strict
from scripts.tracking.trajectory_search import audit_velocity_warning_full_horizon as warning

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/braking_constraint_ablation_20261110.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/braking_constraint_ablation_20261110.md"
QA = REPO / "outputs/qa/braking_constraint_ablation_20261110/result.json"
DRY = REPO / "outputs/braking_constraint_ablation_dry_20261110/result.json"
OUTPUT = REPO / "outputs/braking_constraint_ablation_20261110"
STRICT_RESULT = {
    "path": str(REPO / "outputs/full_horizon_feasibility_restoration_20261110/result.json"),
    "sha256": "8c86d5a94462d2b00bef138026695d0bf716353ab1afebf0f7055d76c02333b2",
}
CASES = (2, 6)


def source_refs():
    paths = (Path(__file__), CONFIG, PLAN, REPO / "tests/research/native_runtime/test_braking_ablation_inputs.py")
    return strict.source_refs() + [prefix.reference(path) for path in paths]


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    c = prefix.read(config_ref)
    prefix.require(
        c
        == {
            "schema": "s2a.braking-constraint-ablation.v1",
            "cases": [2, 6],
            "sampling_seed": 43,
            "warning_steps": [39, 47],
            "chunk_indices": [4, 5],
            "cursor": 7,
            "branches": ["original_seed", "strict_brake_warm"],
            "horizon": 8,
            "control_dt_s": 0.05,
            "inner_dt_s": 0.002,
            "joint_buffer_rad": 0.05,
            "braking_time_s": 0.1,
            "enforce_extra_braking": False,
            "position_cap_m": 0.01,
            "rotation_cap_rad": 0.05,
            "arm_rate_limit": 0.25,
            "max_models": 2,
            "max_forward_queries": 2,
            "max_optimizer_calls": 4,
            "max_optimizer_responses": 1280,
            "max_independent_endpoint_queries": 800,
            "max_independent_transition_steps": 10400,
            "worker_timeout_seconds": 1800,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
        "Exact registered extra-braking ablation",
    )
    recorded = prefix.read(STRICT_RESULT)
    prefix.require(
        recorded["complete"] is recorded["passed"] is recorded["sources_unchanged"] is True,
        "Prior strict-braking restoration closed",
    )
    inputs = strict.load_inputs()
    refs = inputs["source_sha256"]
    refs.update({str(CONFIG): config_ref["sha256"], STRICT_RESULT["path"]: STRICT_RESULT["sha256"]})
    saved = {}
    for ref in recorded["case_refs"]:
        row = prefix.read(ref)
        refs[ref["path"]] = ref["sha256"]
        prefix.require(
            row["complete"] is row["passed"] is True and row["case_index"] in CASES, "Closed strict-braking candidate"
        )
        saved[row["case_index"]] = row["analysis"]
    prefix.require(set(saved) == set(CASES), "One strict-braking candidate per case")
    for case in inputs["cases"]:
        data = saved[case["spec"]["case_index"]]
        prefix.require(
            data["warning"] == case["first_warning"] and data["reference_chunk"] == case["intervention_spec"],
            "Same causal state and generated reference",
        )
        warm = data["candidates"]["restore_joint_warm"]
        verification = warm["independent_verification"]
        prefix.require(
            verification["passed"] is True
            and verification["constraint_minima"]["braking_lower"] >= -1e-10
            and verification["constraint_minima"]["braking_upper"] >= -1e-10,
            "Warm plan has independently verified strict braking",
        )
        case["strict_warm"] = np.asarray(warm["best_joint_safe"]["planned_commands"], dtype=float)
        case["strict_normals"] = np.asarray(warm["best_joint_safe"]["terminal_normals"], dtype=float).reshape(-1, 7)
        case["strict_quality_slack"] = verification["actual_quality_slack"]
    return inputs


def analyze(case, stats):
    current, site_from_tcp, targets, args, differences = warning.restore_warning_context(case, stats)
    profile, output = np.zeros((25, 7)), {}
    for label in ("original_seed", "strict_brake_warm"):
        extra = {"initial_guess": case["strict_warm"]} if label == "strict_brake_warm" else {}
        stats["optimizer_calls_attempted"] += 1
        started = time.monotonic()
        solved = solve_full_horizon(
            *args, past_inner_error_rad=profile, feasibility_restoration=True, enforce_braking=False, **extra
        )
        stats["optimizer_calls_completed"] += 1
        stats["optimizer_responses"] += int(solved["response_evaluations"])
        prefix.require(
            solved["response_evaluations"] <= 320 and solved["braking_constraint_enforced"] is False,
            "Per-search diagnostic scope",
        )
        np.testing.assert_array_equal(solved["terminal_normals"], case["strict_normals"])
        candidate = solved.get("best_joint_safe")
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
                enforce_pose_caps=False,
                enforce_braking=False,
            )
            prefix.require(
                verification["passed"] is True
                and verification["independent_endpoint_queries"] == 200
                and verification["independent_transition_steps"] == 2600
                and verification["braking_constraint_enforced"] is False,
                "Bound independent diagnostic audit",
            )
            e = verification["tcp_errors"]
            actual_slack = max(
                0.0,
                float(
                    np.max(
                        np.r_[np.linalg.norm(e[:, :3], axis=1) / 0.01 - 1, np.linalg.norm(e[:, 3:], axis=1) / 0.05 - 1]
                    )
                ),
            )
            np.testing.assert_allclose(actual_slack, candidate["quality_slack"], atol=1e-8, rtol=0)
            verification["actual_quality_slack"] = actual_slack
        if solved["witness_found"]:
            prefix.require(
                verification is not None and verification["all_pose_caps_satisfied"] is True,
                "All reported model quality witnesses meet original pose caps",
            )
        summary = {k: v for k, v in solved.items() if k not in ("rollout", "best_joint_safe")}
        if candidate is not None:
            summary["best_joint_safe"] = {k: v for k, v in candidate.items() if k != "rollout"}
        summary.update(
            independent_verification=verification,
            ablated_quality_witness=bool(verification and verification["all_pose_caps_satisfied"]),
            strict_braking_also_satisfied=bool(verification and verification["braking_constraint_satisfied"]),
            elapsed_seconds=time.monotonic() - started,
        )
        output[label] = summary
    return {
        "case_index": case["spec"]["case_index"],
        "warning": case["first_warning"],
        "reference_chunk": case["intervention_spec"],
        "reconstruction_max_difference": differences,
        "strict_quality_slack": case["strict_quality_slack"],
        "candidates": output,
        "scope": "static ablation only; no dispatch or physical qualification",
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
            and stats["optimizer_calls_completed"] == 2
            and stats["optimizer_responses"] <= 640
            and stats["independent_endpoint_queries_charged"] <= 400
            and stats["independent_transition_steps_charged"] <= 5200,
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
        dry["complete"] is dry["passed"] is True and dry["models"] == dry["physics_steps"] == 0, "Zero-model dry closed"
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
        "Dry source/input identity",
    )
    registration = publish(
        OUTPUT / "registration.json",
        {
            "dry_ref": dry_ref,
            "qa_ref": qa_ref,
            "strict_result_ref": STRICT_RESULT,
            "source_refs": sources,
            "source_sha256": inputs["source_sha256"],
            "asset_refs": inputs["asset_refs"],
            "cases": list(CASES),
            "max_models": 2,
            "max_forward_queries": 2,
            "max_optimizer_calls": 4,
            "max_optimizer_responses": 1280,
            "max_independent_endpoint_queries": 800,
            "max_independent_transition_steps": 10400,
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
                "scripts.tracking.trajectory_search.audit_braking_constraint_ablation",
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
                w["returncode"] == 0 and w["actual_wait"] and w["reaped"] and not w["signal_sent"]
                for w in lifecycle["workers"]
            ),
            "Both workers closed and reaped",
        )
        refs = [prefix.reference(OUTPUT / f"case_{index:02d}" / "result.json") for index in CASES]
        rows = [prefix.read(ref) for ref in refs]
        prefix.require(all(row["complete"] is row["passed"] is True for row in rows), "Complete ablation audits")
        prefix.require(
            all(prefix.reference(path)["sha256"] == digest for path, digest in inputs["source_sha256"].items())
            and source_refs() == sources
            and snapshot(REPO)["sources"] == before,
            "Sources and inputs unchanged",
        )
        totals = {
            name: sum(row["stats"][name] for row in rows)
            for name in (
                "model_initializations_completed",
                "forward_queries_completed",
                "baseline_rollouts",
                "optimizer_calls_completed",
                "optimizer_responses",
                "independent_endpoint_queries_charged",
                "independent_transition_steps_charged",
            )
        }
        prefix.require(
            totals["model_initializations_completed"] == totals["forward_queries_completed"] == 2
            and totals["baseline_rollouts"] == 2
            and totals["optimizer_calls_completed"] == 4
            and totals["optimizer_responses"] <= 1280
            and totals["independent_endpoint_queries_charged"] <= 800
            and totals["independent_transition_steps_charged"] <= 10400,
            "Registered aggregate budget",
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=refs, **totals)
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", type=int, choices=CASES)
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
                    "cases": list(CASES),
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
        worker(args.worker)


if __name__ == "__main__":
    main()
